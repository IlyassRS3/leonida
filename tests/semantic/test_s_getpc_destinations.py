"""Semantic tests for `s_getpc_b64` destinations (`vcc`, 2-slot `slist`, `null`) and downstream uses.

RDNA2 ISA SOP1 `s_getpc_b64` (opcode 0x1f) writes the 64-bit address of the next instruction into a 64-bit
scalar destination (`SDST`, LLVM `SReg_64`):
  - `s[a:a+1]` (2-SGPR pair, 0..104)
  - `vcc` (`s[106:107]`, `{vcc_lo, vcc_hi}` -> `%vcc` / `%vcc_hi`)
  - `[s105,vcc_lo]` (2-slot `slist` straddling `s105` and `vcc_lo`, `sdst=105`)
  - `null` (`sdst=125`, result discarded, no `uses_codebase`, still requires `; encoding:` bytes)

Covers:
  - 64-bit code-base + next-PC addition (including 32-bit carry into the high word) into `vcc`, `[s105,vcc_lo]`,
    and `null` across wave32, wave64, and `--wave64-emulate`
  - `s_getpc_b64 vcc` / `[s105,vcc_lo]` feeding `s_add_u32` + `s_addc_u32` + `s_load_dword` and `global_load_dword`
  - Prior wave32 `vcc_hi` and wave64 `vcc` lane masks cleared so subsequent loads read raw PC words
  - `s_getpc_b64 vcc` feeding `s_mov_b64 s[a:b], vcc` (directly and after `s_add_u32`/`s_addc_u32` half updates)
    via `wave64_emulate_opt.mark_vcc_data_moves`
  - Far branches (`fold_far_branches`) using `vcc` and `[s105,vcc_lo]` (forward `add`/`addc` and backward `sub`/`subb`)
  - Wave32 FSSM header and sidecar-less wave-size inference (`s_getpc_b64 vcc` is not a wave64 marker)
  - Rejection of unsupported destinations (`exec`, `m0`, `vcc_lo`, `vcc_hi`, `ttmp[0:1]`, `s4`, `s[4:7]`, `v[0:1]`)
    and missing `; encoding:` bytes
"""
import gc
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
for p in (HERE, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import rdna2_decode  # noqa: E402
import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]


def _enc(nbytes=4):
    return "; encoding: [" + ",".join(["0x00"] * nbytes) + "]"


def _fssm(wave_size, cs=False):
    blob = bytearray(0x138)
    blob[0:4] = b"FSSM"
    if wave_size == 32:
        struct.pack_into("<I", blob, 0xD4, 0x20)
    if cs:
        return bytes(blob) + struct.pack("<4I", 0x8088, wave_size, 1, 1)
    return bytes(blob)


def _validate_spirv(name, spv):
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not (spirv_as and spirv_val):
        return
    with tempfile.TemporaryDirectory() as tmp:
        asm_p = os.path.join(tmp, "s.spvasm")
        spv_p = os.path.join(tmp, "s.spv")
        with open(asm_p, "w", encoding="utf-8", newline="\n") as f:
            f.write(spv)
        r1 = subprocess.run([spirv_as, "--target-env", "vulkan1.2", asm_p, "-o", spv_p],
                            capture_output=True, text=True)
        if r1.returncode != 0:
            raise AssertionError(f"spirv-as failed for {name}:\n{r1.stderr}")
        r2 = subprocess.run([spirv_val, "--target-env", "vulkan1.2", spv_p],
                            capture_output=True, text=True)
        if r2.returncode != 0:
            raise AssertionError(f"spirv-val failed for {name}:\n{r2.stderr}")


def _translate_ps(name, lines, wave_size=None, emulate=False):
    prev_emu = rdna2spirv.WAVE64_EMULATE
    try:
        rdna2spirv.WAVE64_EMULATE = emulate
        fssm = None if wave_size is None else _fssm(wave_size, cs=False)
        spv = support.translate_lines("ps_" + name, lines, fssm=fssm)
        gen = rdna2spirv.translate.last_gen
        _validate_spirv("ps_" + name, spv)
        return spv, gen
    finally:
        rdna2spirv.WAVE64_EMULATE = prev_emu


def _translate_cs(name, lines, wave_size=32, emulate=False):
    prev_emu = rdna2spirv.WAVE64_EMULATE
    try:
        rdna2spirv.WAVE64_EMULATE = emulate
        with tempfile.TemporaryDirectory() as tmp:
            os.mkdir(os.path.join(tmp, "disasm"))
            os.mkdir(os.path.join(tmp, "fssm"))
            s_path = os.path.join(tmp, "disasm", "cs_" + name + ".s")
            with open(s_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(f"; cs_{name}\n")
                for ln in lines:
                    f.write(f"\t{ln}\n")
            with open(os.path.join(tmp, "cs_" + name + ".dxbc"), "wb") as f:
                f.write(STUB_DXBC)
            with open(os.path.join(tmp, "fssm", "cs_" + name + ".fssm"), "wb") as f:
                f.write(_fssm(wave_size, cs=True))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(s_path)
                gen = rdna2spirv.translate.last_gen
                gc.collect()
        _validate_spirv("cs_" + name, spv)
        return spv, gen
    finally:
        rdna2spirv.WAVE64_EMULATE = prev_emu


def _translate_cs_fspx(name, lines, wave32=True):
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        os.mkdir(os.path.join(tmp, "fspx"))
        with open(os.path.join(tmp, "fspx", "cs_" + name + ".json"), "w", encoding="utf-8", newline="\n") as f:
            json.dump({
                "threads": [32 if wave32 else 64, 1, 1],
                "wave32": wave32,
                "user_sgprs": 8,
                "tgid_enable": [1, 0, 0],
                "tidig_comp_cnt": 0,
                "lds_bytes": 0,
                "rsrc2": 0x90,
            }, f)
        s_path = os.path.join(tmp, "disasm", "cs_" + name + ".s")
        with open(s_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; cs_{name}\n")
            for ln in lines:
                f.write(f"\t{ln}\n")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(s_path)
            gen = rdna2spirv.translate.last_gen
            gc.collect()
    _validate_spirv("cs_" + name, spv)
    return spv, gen


def _pc_with_codebase(code_base, user_sgprs=None):
    pc = list(user_sgprs or [0] * 32)
    if len(pc) < 32:
        pc += [0] * (32 - len(pc))
    pc = pc[:32] + [0, 0, code_base & M32, (code_base >> 32) & M32]
    return pc


def _run_cs_words(spv, wave_size, emulate, code_base, memory=None):
    mem = dict(memory or {})
    sg_size = 32 if emulate else wave_size
    pc = _pc_with_codebase(code_base, VSHARP)
    si.run_workgroup(
        spv, wave_size, sg_size,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % sg_size},
        push_constants=pc, memory=mem,
    )
    return mem


class GetPcDestinationTests(unittest.TestCase):
    def test_vcc_destination_carries_across_4gb_boundary(self):
        # s_nop 0 (4B at pc=0) + s_getpc_b64 vcc (4B at pc=4) -> next PC is code_base + 8.
        code_base = 0x00000005_FFFFFFFC
        expected = code_base + 8  # 0x00000006_00000004 (tests carry from lo to hi)
        lines = [
            f"s_nop 0 {_enc(4)}",
            f"s_getpc_b64 vcc ; encoding: [0x00,0x1f,0xea,0xbe]",
            f"v_mov_b32_e32 v4, vcc_lo {_enc(4)}",
            f"v_mov_b32_e32 v5, vcc_hi {_enc(4)}",
            f"v_mov_b32_e32 v6, 0 {_enc(4)}",
            f"v_mov_b32_e32 v7, 0 {_enc(4)}",
            f"exp mrt0 v4, v5, v6, v7 done vm {_enc(8)}",
            f"s_endpgm {_enc(4)}",
        ]
        for label, wave_size in (("sidecarless", None), ("fssm_w32", 32), ("fssm_w64", 64)):
            with self.subTest(mode=label):
                spv, gen = _translate_ps(f"getpc_vcc_{label}", lines, wave_size=wave_size)
                self.assertEqual(gen.wave64, wave_size == 64)
                self.assertTrue(gen.uses_codebase)
                words = support.mrt0_words(spv, push_constants=_pc_with_codebase(code_base))
                self.assertEqual(words[:2], [expected & M32, (expected >> 32) & M32])

    def test_vcc_destination_feeds_sload_and_clears_prior_masks(self):
        # Prior lane masks live in vcc_lo (and vcc_hi in wave32); s_getpc_b64 vcc must overwrite both with raw PC.
        code_base = 0x00000003_00010000
        # pc=0: v_cmp_lt_u32 (8B) -> pc=8: v_cmp_lt_u32_e64 vcc_hi (8B in w32) or s_nop (4B in w64)
        for wave_size, emulate in ((32, False), (64, False), (64, True)):
            with self.subTest(wave_size=wave_size, emulate=emulate):
                if wave_size == 32:
                    setup = [
                        f"v_cmp_lt_u32_e32 vcc_lo, 5, v0 {_enc(4)}",
                        f"v_cmp_lt_u32_e64 vcc_hi, v0, 12 {_enc(8)}",
                    ]
                    nxt = 16  # 4 + 8 + 4
                else:
                    setup = [
                        f"v_cmp_lt_u32_e32 vcc, 5, v0 {_enc(4)}",
                        f"s_nop 0 {_enc(4)}",
                    ]
                    nxt = 12  # 4 + 4 + 4
                lines = setup + [
                    f"s_getpc_b64 vcc ; encoding: [0x00,0x1f,0xea,0xbe]",
                    f"s_add_u32 vcc_lo, vcc_lo, 0x40 {_enc(8)}",
                    f"s_addc_u32 vcc_hi, vcc_hi, 0 {_enc(4)}",
                    f"s_load_dword s10, vcc, 0x8 {_enc(8)}",
                    f"s_waitcnt lgkmcnt(0) {_enc(4)}",
                    f"v_mov_b32_e32 v4, s10 {_enc(4)}",
                    f"v_lshlrev_b32_e32 v5, 2, v0 {_enc(4)}",
                    f"buffer_store_dword v4, v5, s[0:3], 0 offen {_enc(8)}",
                    f"s_endpgm {_enc(4)}",
                ]
                spv, gen = _translate_cs(f"getpc_vcc_sload_{wave_size}_{int(emulate)}", lines,
                                         wave_size=wave_size, emulate=emulate)
                self.assertEqual(gen.wave64, wave_size == 64)
                target_addr = code_base + nxt + 0x40 + 0x8
                mem = _run_cs_words(spv, wave_size, emulate, code_base, memory={target_addr: 0xDEADBEEF})
                self.assertEqual([mem[4 * i] for i in range(wave_size)], [0xDEADBEEF] * wave_size)

    def test_vcc_destination_feeds_global_load(self):
        code_base = 0x00000004_00020000
        lines = [
            f"s_getpc_b64 vcc ; encoding: [0x00,0x1f,0xea,0xbe]",
            f"s_add_u32 vcc_lo, vcc_lo, 0x100 {_enc(8)}",
            f"s_addc_u32 vcc_hi, vcc_hi, 0 {_enc(4)}",
            f"v_lshlrev_b32_e32 v5, 2, v0 {_enc(4)}",
            f"global_load_dword v4, v5, vcc {_enc(8)}",
            f"s_waitcnt vmcnt(0) {_enc(4)}",
            f"buffer_store_dword v4, v5, s[0:3], 0 offen {_enc(8)}",
            f"s_endpgm {_enc(4)}",
        ]
        table_base = code_base + 4 + 0x100
        for wave_size, emulate in ((32, False), (64, False), (64, True)):
            with self.subTest(wave_size=wave_size, emulate=emulate):
                spv, _ = _translate_cs(f"getpc_vcc_global_{wave_size}_{int(emulate)}", lines,
                                       wave_size=wave_size, emulate=emulate)
                seed = {table_base + 4 * i: 0xA0000000 + i for i in range(wave_size)}
                mem = _run_cs_words(spv, wave_size, emulate, code_base, memory=seed)
                self.assertEqual([mem[4 * i] for i in range(wave_size)], [0xA0000000 + i for i in range(wave_size)])

    def test_vcc_destination_feeds_s_mov_b64_pair_copy(self):
        # Both direct `s_getpc_b64 vcc; s_mov_b64 s[6:7], vcc` and half-updated `s_add_u32`/`s_addc_u32`
        # before `s_mov_b64 s[6:7], vcc` must preserve both 32-bit words (not replicate vcc_lo into s7).
        code_base = 0x00000007_FFFFF000
        for with_add in (False, True):
            delta = 0x2000 if with_add else 0
            expected = code_base + 4 + delta
            mid = (
                [f"s_add_u32 vcc_lo, vcc_lo, {hex(delta)} {_enc(8)}",
                 f"s_addc_u32 vcc_hi, vcc_hi, 0 {_enc(4)}"]
                if with_add else []
            )
            lines = [
                f"s_getpc_b64 vcc ; encoding: [0x00,0x1f,0xea,0xbe]",
                *mid,
                f"s_mov_b64 s[6:7], vcc {_enc(4)}",
                f"v_mov_b32_e32 v4, s6 {_enc(4)}",
                f"v_mov_b32_e32 v5, s7 {_enc(4)}",
                f"v_lshlrev_b32_e32 v2, 3, v0 {_enc(4)}",
                f"buffer_store_dwordx2 v[4:5], v2, s[0:3], 0 offen {_enc(8)}",
                f"s_endpgm {_enc(4)}",
            ]
            for wave_size, emulate in ((64, False), (64, True)):
                with self.subTest(with_add=with_add, wave_size=wave_size, emulate=emulate):
                    spv, _ = _translate_cs(f"getpc_vcc_mov64_{int(with_add)}_{wave_size}_{int(emulate)}", lines,
                                           wave_size=wave_size, emulate=emulate)
                    mem = _run_cs_words(spv, wave_size, emulate, code_base)
                    self.assertEqual((mem[0], mem[4]), (expected & M32, (expected >> 32) & M32))

        # FSPX wave32 also allows s_mov_b64 s[6:7], vcc after s_getpc_b64 vcc and after s_sub_u32/s_subb_u32.
        sub_base = 0x00000008_00000010
        sub_lines = [
            f"s_getpc_b64 vcc ; encoding: [0x00,0x1f,0xea,0xbe]",
            f"s_sub_u32 vcc_lo, vcc_lo, 0x20 {_enc(8)}",
            f"s_subb_u32 vcc_hi, vcc_hi, 0 {_enc(4)}",
            f"s_mov_b64 s[6:7], vcc {_enc(4)}",
            f"v_mov_b32_e32 v4, s6 {_enc(4)}",
            f"v_mov_b32_e32 v5, s7 {_enc(4)}",
            f"v_lshlrev_b32_e32 v2, 3, v0 {_enc(4)}",
            f"buffer_store_dwordx2 v[4:5], v2, s[0:3], 0 offen {_enc(8)}",
            f"s_endpgm {_enc(4)}",
        ]
        spv_w32, gen_w32 = _translate_cs_fspx("getpc_vcc_sub_mov64_w32", sub_lines, wave32=True)
        self.assertFalse(gen_w32.wave64)
        mem_w32 = _run_cs_words(spv_w32, 32, False, sub_base)
        exp_sub = sub_base + 4 - 0x20  # 0x00000007_FFFFFFF4 (borrows across 4 GiB boundary)
        self.assertEqual((mem_w32[0], mem_w32[4]), (exp_sub & M32, (exp_sub >> 32) & M32))

    def test_vcc_destination_wave64_vccnz_checks_high_word_when_lo_wraps_zero(self):
        # code_base + 4 == 0x00000006_00000000: vcc_lo is 0, vcc_hi is 6.
        # Because s_getpc_b64 vcc clears the wave64 lane-mask tag, s_cbranch_vccnz must see vcc_hi != 0 and jump.
        code_base = 0x00000005_FFFFFFFC
        lines = [
            f"v_cmp_eq_u32_e32 vcc, 0, v0 {_enc(4)}",
            f"s_getpc_b64 vcc ; encoding: [0x00,0x1f,0xea,0xbe]",
            f"s_mov_b32 s10, 7 {_enc(4)}",
            f"s_cbranch_vccnz 1 {_enc(4)}",
            f"s_mov_b32 s10, 9 {_enc(4)}",
            f"v_mov_b32_e32 v4, s10 {_enc(4)}",
            f"v_lshlrev_b32_e32 v5, 2, v0 {_enc(4)}",
            f"buffer_store_dword v4, v5, s[0:3], 0 offen {_enc(8)}",
            f"s_endpgm {_enc(4)}",
        ]
        # pc=0: v_cmp (4B), pc=4: s_getpc_b64 (4B) -> next PC is 8; pick code_base so code_base + 8 == 0x6_00000000.
        spv, gen = _translate_cs("getpc_vcc_branch_hi", lines, wave_size=64, emulate=False)
        self.assertTrue(gen.wave64)
        mem = _run_cs_words(spv, 64, False, code_base - 4)
        self.assertEqual([mem[4 * i] for i in range(64)], [7] * 64)

    def test_global_load_vcc_lane_mask_pointer_refused_in_emulated_wave64(self):
        lines = [
            f"v_cmp_lt_u32_e32 vcc, 5, v0 {_enc(4)}",
            f"v_lshlrev_b32_e32 v5, 2, v0 {_enc(4)}",
            f"global_load_dword v4, v5, vcc {_enc(8)}",
            f"s_waitcnt vmcnt(0) {_enc(4)}",
            f"buffer_store_dword v4, v5, s[0:3], 0 offen {_enc(8)}",
            f"s_endpgm {_enc(4)}",
        ]
        with self.assertRaisesRegex(rdna2spirv.Unsupported, "emulated wave64 lane-mask pointer"):
            _translate_cs("global_vcc_mask_emu", lines, wave_size=64, emulate=True)

    def test_pc_tail_ubo_flag_for_vcc_vs_null(self):
        prev_ubo = rdna2spirv.PC_TAIL_UBO
        try:
            rdna2spirv.PC_TAIL_UBO = True
            vcc_spv, vcc_gen = _translate_ps("getpc_vcc_ubo", [
                f"s_getpc_b64 vcc ; encoding: [0x00,0x1f,0xea,0xbe]",
                f"v_mov_b32_e32 v4, vcc_lo {_enc(4)}",
                f"exp mrt0 v4, v4, v4, v4 done vm {_enc(8)}",
                f"s_endpgm {_enc(4)}",
            ], wave_size=32)
            self.assertTrue(vcc_gen.uses_codebase)
            self.assertTrue(vcc_gen.uses_pc_tail)
            self.assertIn("%pc_tail", vcc_spv)

            null_spv, null_gen = _translate_ps("getpc_null_ubo", [
                f"s_getpc_b64 null ; encoding: [0x00,0x1f,0xfd,0xbe]",
                f"v_mov_b32_e32 v4, 1 {_enc(4)}",
                f"exp mrt0 v4, v4, v4, v4 done vm {_enc(8)}",
                f"s_endpgm {_enc(4)}",
            ], wave_size=32)
            self.assertFalse(null_gen.uses_codebase)
            self.assertFalse(null_gen.uses_pc_tail)
            self.assertNotIn("%pc_tail", null_spv)
        finally:
            rdna2spirv.PC_TAIL_UBO = prev_ubo

    def test_slist_destination_writes_s105_and_vcc_lo_preserving_vcc_hi(self):
        # sdst=105 decodes to [s105, vcc_lo]; vcc_hi must stay untouched.
        dec = rdna2_decode.decode_bytes(struct.pack("<I", 0xBEE91F00))["insns"]
        self.assertEqual(len(dec), 1)
        self.assertEqual(dec[0]["op"], "s_getpc_b64")
        self.assertEqual(dec[0]["args"], [{"k": "slist", "r": [105, "vcc_lo"], "n": 2}])

        code_base = 0x00000009_FFFFFFFC
        expected = code_base + 8  # s_mov_b32 vcc_hi (4B at pc=0) + s_getpc_b64 (4B at pc=4) -> nxt=8
        lines = [
            f"s_mov_b32 vcc_hi, 0x13572468 {_enc(4)}",
            f"s_getpc_b64 [s105,vcc_lo] ; encoding: [0x00,0x1f,0xe9,0xbe]",
            f"v_mov_b32_e32 v4, s105 {_enc(4)}",
            f"v_mov_b32_e32 v5, vcc_lo {_enc(4)}",
            f"v_mov_b32_e32 v6, vcc_hi {_enc(4)}",
            f"v_mov_b32_e32 v7, 0 {_enc(4)}",
            f"exp mrt0 v4, v5, v6, v7 done vm {_enc(8)}",
            f"s_endpgm {_enc(4)}",
        ]
        spv, gen = _translate_ps("getpc_slist_105", lines, wave_size=32)
        self.assertFalse(gen.wave64)
        self.assertTrue(gen.uses_codebase)
        words = support.mrt0_words(spv, push_constants=_pc_with_codebase(code_base))
        self.assertEqual(words[:3], [expected & M32, (expected >> 32) & M32, 0x13572468])

        # Also check [s105,vcc_lo] feeding s_load_dword.
        table_base = 0x00000002_00008000
        sload_lines = [
            f"s_getpc_b64 [s105,vcc_lo] ; encoding: [0x00,0x1f,0xe9,0xbe]",
            f"s_add_u32 s105, s105, 0x80 {_enc(8)}",
            f"s_addc_u32 vcc_lo, vcc_lo, 0 {_enc(4)}",
            f"s_load_dword s10, [s105,vcc_lo], 0x4 {_enc(8)}",
            f"s_waitcnt lgkmcnt(0) {_enc(4)}",
            f"v_mov_b32_e32 v4, s10 {_enc(4)}",
            f"exp mrt0 v4, v4, v4, v4 done vm {_enc(8)}",
            f"s_endpgm {_enc(4)}",
        ]
        spv2, _ = _translate_ps("getpc_slist_sload", sload_lines, wave_size=32)
        got = support.mrt0_words(spv2, push_constants=_pc_with_codebase(table_base),
                                 memory={table_base + 4 + 0x80 + 0x4: 0xFEEDFACE})
        self.assertEqual(got, [0xFEEDFACE] * 4)

    def test_null_destination_discards_result_without_codebase(self):
        lines = [
            f"s_mov_b32 vcc_lo, 0x11112222 {_enc(8)}",
            f"s_mov_b32 vcc_hi, 0x33334444 {_enc(8)}",
            f"s_getpc_b64 null ; encoding: [0x00,0x1f,0xfd,0xbe]",
            f"v_mov_b32_e32 v4, vcc_lo {_enc(4)}",
            f"v_mov_b32_e32 v5, vcc_hi {_enc(4)}",
            f"v_mov_b32_e32 v6, 0 {_enc(4)}",
            f"v_mov_b32_e32 v7, 0 {_enc(4)}",
            f"exp mrt0 v4, v5, v6, v7 done vm {_enc(8)}",
            f"s_endpgm {_enc(4)}",
        ]
        spv, gen = _translate_ps("getpc_null", lines, wave_size=32)
        self.assertFalse(gen.uses_codebase)
        self.assertNotIn("%ud_code_lo_p", spv)
        words = support.mrt0_words(spv)
        self.assertEqual(words[:2], [0x11112222, 0x33334444])

    def test_far_branch_using_vcc_and_slist(self):
        # Forward far branch over a poison write, plus backward loop via s_sub_u32 / s_subb_u32 using vcc.
        # Byte offsets:
        #   0:  s_mov_b32 s10, 0            (4B)
        #   4:  s_getpc_b64 vcc             (4B) -> next PC = 8
        #   8:  s_add_u32 vcc_lo, vcc_lo, 20 (8B) -> target = 8 + 20 = 28
        #   16: s_addc_u32 vcc_hi, vcc_hi, 0 (4B)
        #   20: s_setpc_b64 vcc             (4B)
        #   24: s_mov_b32 s10, 99           (4B)  [skipped poison]
        #   28: s_add_u32 s10, s10, 1       (4B)  [loop head]
        #   32: s_cmp_lt_u32 s10, 3         (4B)
        #   36: s_cbranch_scc0 5            (4B)  -> next=40, +5*4=20 -> target=60
        #   40: s_getpc_b64 vcc             (4B)  -> next PC = 44
        #   44: s_sub_u32 vcc_lo, vcc_lo, 16 (8B) -> target = 44 - 16 = 28
        #   52: s_subb_u32 vcc_hi, vcc_hi, 0 (4B)
        #   56: s_setpc_b64 vcc             (4B)
        #   60: v_mov_b32_e32 v4, s10       (4B)
        #   64: exp mrt0 v4, v4, v4, v4 done vm (8B)
        #   72: s_endpgm                    (4B)
        vcc_lines = [
            f"s_mov_b32 s10, 0 {_enc(4)}",
            f"s_getpc_b64 vcc ; encoding: [0x00,0x1f,0xea,0xbe]",
            f"s_add_u32 vcc_lo, vcc_lo, 20 {_enc(8)}",
            f"s_addc_u32 vcc_hi, vcc_hi, 0 {_enc(4)}",
            f"s_setpc_b64 vcc ; encoding: [0x6a,0x1d,0x80,0xbe]",
            f"s_mov_b32 s10, 99 {_enc(4)}",
            f"s_add_u32 s10, s10, 1 {_enc(4)}",
            f"s_cmp_lt_u32 s10, 3 {_enc(4)}",
            f"s_cbranch_scc0 5 {_enc(4)}",
            f"s_getpc_b64 vcc ; encoding: [0x00,0x1f,0xea,0xbe]",
            f"s_sub_u32 vcc_lo, vcc_lo, 16 {_enc(8)}",
            f"s_subb_u32 vcc_hi, vcc_hi, 0 {_enc(4)}",
            f"s_setpc_b64 vcc ; encoding: [0x6a,0x1d,0x80,0xbe]",
            f"v_mov_b32_e32 v4, s10 {_enc(4)}",
            f"exp mrt0 v4, v4, v4, v4 done vm {_enc(8)}",
            f"s_endpgm {_enc(4)}",
        ]
        for wave_size in (32, 64):
            with self.subTest(kind="vcc", wave_size=wave_size):
                spv, gen = _translate_ps(f"far_vcc_w{wave_size}", vcc_lines, wave_size=wave_size)
                self.assertEqual(gen.wave64, wave_size == 64)
                self.assertEqual(support.mrt0_words(spv), [3, 3, 3, 3])

        # Forward far branch using [s105,vcc_lo].
        slist_lines = [
            f"s_mov_b32 s10, 7 {_enc(4)}",
            f"s_getpc_b64 [s105,vcc_lo] ; encoding: [0x00,0x1f,0xe9,0xbe]",
            f"s_add_u32 s105, s105, 20 {_enc(8)}",
            f"s_addc_u32 vcc_lo, vcc_lo, 0 {_enc(4)}",
            f"s_setpc_b64 [s105,vcc_lo] {_enc(4)}",
            f"s_mov_b32 s10, 99 {_enc(4)}",
            f"v_mov_b32_e32 v4, s10 {_enc(4)}",
            f"exp mrt0 v4, v4, v4, v4 done vm {_enc(8)}",
            f"s_endpgm {_enc(4)}",
        ]
        spv_s, _ = _translate_ps("far_slist", slist_lines, wave_size=32)
        self.assertEqual(support.mrt0_words(spv_s), [7, 7, 7, 7])

    def test_invalid_destinations_and_missing_encoding_are_rejected(self):
        for bad_dst in ("exec", "m0", "vcc_lo", "vcc_hi", "ttmp[0:1]", "s4", "s[4:7]", "v[0:1]"):
            with self.subTest(bad_dst=bad_dst):
                lines = [
                    f"s_getpc_b64 {bad_dst} {_enc(4)}",
                    f"v_mov_b32_e32 v4, 0 {_enc(4)}",
                    f"exp mrt0 v4, v4, v4, v4 done vm {_enc(8)}",
                    f"s_endpgm {_enc(4)}",
                ]
                with self.assertRaisesRegex(rdna2spirv.Unsupported, "s_getpc_b64 destination"):
                    _translate_ps("getpc_bad_dst", lines, wave_size=32)

        for dst in ("vcc", "[s105,vcc_lo]", "null"):
            with self.subTest(missing_enc_dst=dst):
                lines = [
                    f"s_getpc_b64 {dst}",
                    "v_mov_b32_e32 v4, 0",
                    "exp mrt0 v4, v4, v4, v4 done vm",
                    "s_endpgm",
                ]
                with self.assertRaisesRegex(rdna2spirv.Unsupported, "s_getpc_b64 without encoding"):
                    _translate_ps("getpc_noenc", lines, wave_size=32)


if __name__ == "__main__":
    unittest.main()
