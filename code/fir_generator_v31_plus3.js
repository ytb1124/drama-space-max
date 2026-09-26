// fir_generator_v31_plus3.js
// v31_plus3 파이썬 스크립트 실행 + 결과 FIR 경로 전달
// 변경점: 5번 인자(out)가 파일이든 폴더든, 6번 인자(name)가 있으면 그 이름을 "항상" 사용

const Max = require("max-api");
const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

Max.post("fir_generator loaded");
Max.outlet("ready");

const bad  = (p)=>!p || p==='s' || p==='text' || typeof p!=="string";
const isAbs= (p)=>typeof p==="string" && p[0]==='/';

function sanitizeName(name){
  let n = String(name||"").trim();
  if (!n) n = "club2hall_fir_v31_plus3";
  n = n.replace(/\.[^/.]+$/,"");          // 확장자 제거
  n = n.replace(/[^A-Za-z0-9_\-]+/g,"_"); // 안전 문자만
  return n + ".wav";
}

/**
 * outArg: 폴더 or 파일 경로(절대경로 권장)
 * nameArg: 파일명(확장자 없이 넣어도 됨)
 * 규칙:
 *  - outArg가 '폴더'면: join(folder, sanitize(nameArg))
 *  - outArg가 '파일'이어도: nameArg가 있으면 해당 이름으로 "강제 치환"
 *  - nameArg가 비어있으면: outArg가 파일이면 그걸 사용(확장자 보정), 폴더면 기본 이름 사용
 */
function makeOutPath(outArg, nameArg){
  if (bad(outArg)) return null;
  const p = String(outArg);

  // 폴더
  if (fs.existsSync(p) && fs.statSync(p).isDirectory()){
    return path.join(p, sanitizeName(nameArg));
  }

  // 파일 경로
  const dir = path.dirname(p);
  if (!fs.existsSync(dir)) return null;

  if (nameArg && String(nameArg).trim()){
    return path.join(dir, sanitizeName(nameArg));
  }

  // 이름이 없으면 outArg 그대로(.wav 보정)
  let f = p;
  if (!/\.(wav|wave)$/i.test(f)) f += ".wav";
  return f;
}

Max.addHandler("run", (py, script, club, hall, outArg, nameArg) => {
  // 인자 검증
  if ([py,script,club,hall,outArg].some(bad) || ![py,script,club,hall].every(isAbs)){
    Max.outlet(["error",
      `Run blocked: invalid paths\npy:${py}\nsc:${script}\nclub:${club}\nhall:${hall}\nout:${outArg}`]);
    return;
  }

  const outPath = makeOutPath(outArg, nameArg);
  if (!outPath){
    Max.outlet(["error", `Invalid output path (folder or file): ${outArg}, name: ${nameArg||""}`]);
    return;
  }

  const args = [
    String(script),
    "--club", String(club),
    "--hall", String(hall),
    "--out",  String(outPath),
    "--mode","mono",
    "--band_low","125","--band_high","4000",
    "--transition_hz","900",
    "--reg_db","-32","--smooth_bins","121",
    "--gate_db","-33","--gate_pre_ms","4","--gate_tail_ms","16","--gate_fade_ms","5",
    "--early_ms","40","--xover_ms","16","--late_ms","200",
    "--reg_db_late","-28","--smooth_bins_late","129",
    "--late_phase","keep",
    "--final_align","1",
    "--peak_limit_ir","0.95",
    "--export_full","1"
  ];

  const child = spawn(String(py), args, { shell:false });
  let out = "", err = "";

  child.stdout.on("data", d => out += d.toString());
  child.stderr.on("data", d => err += d.toString());

  child.on("close", code => {
    const text = (out + "\n" + err).trim();
    if (code !== 0 || /^usage:/mi.test(text)) {
      Max.outlet(["error", text || `python exited ${code}`]);
      return;
    }
    try {
      const j = JSON.parse(text);
      if (j && j.error){ Max.outlet(["error", text]); return; }

      Max.outlet(["done", outPath]);  // 최종 파일 경로

      if (j.CorrCoef_early40ms !== undefined)
        Max.outlet(["corrcoef", j.CorrCoef_early40ms]);

      const bands = ["125","250","500","1000","2000","4000"];
      if (j.CohBands_Early_0_40ms)
        for (const k of bands)
          if (j.CohBands_Early_0_40ms[k] !== undefined)
            Max.outlet([`cohE_${k}`, j.CohBands_Early_0_40ms[k]]);
      if (j.CohBands_Late_60_200ms)
        for (const k of bands)
          if (j.CohBands_Late_60_200ms[k] !== undefined)
            Max.outlet([`cohL_${k}`, j.CohBands_Late_60_200ms[k]]);

      Max.outlet(["json", text]);
    } catch(e){
      Max.outlet(["error", "JSON parse failed\n"+text]);
    }
  });
});