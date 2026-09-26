// hybrid_downmix_runner.js  (pack 5개 인렛용)
// RUN 인자 순서(왼→오): <python> <script> <outFolderOrFile> <inIR> <name>
// - 1번 인렛: python 경로(문자열)도 받고, RUN bang도 받는 "핫 인렛"으로 사용
// - outFolderOrFile: 폴더 또는 파일 절대경로(POSIX). 폴더면 name으로 파일명 생성.
// - name: 확장자 없이 파일명(공백/한글 OK). 주어지면 "항상" 이 이름으로 저장.
//
// 파라미터 바꾸기: set <key> <value>
//   keys = mode, early_ms, late_start_ms, xfade_ms, target_sr, peak_limit, ch, weights
//
// 출력:
//   ["ready"]
//   ["param", key, val]
//   ["runargs", py, script, inIR, outPath]
//   ["done", outPath]
//   ["json", jsonText]
//   ["error", text]

const Max = require("max-api");
const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");

const bad  = (p)=>!p || p==='s' || p==='text' || typeof p!=="string";
const isAbs= (p)=>typeof p==="string" && p[0]==='/';

let P = {
  mode: "hybrid",
  early_ms: 40,
  late_start_ms: 60,
  xfade_ms: 10,
  target_sr: 48000,
  peak_limit: 0.98,
  ch: 0,
  weights: ""   // "0.4,0.4,0.1,0.1" 등
};

Max.post("hybrid_downmix_runner (5-arg) loaded");
Max.outlet("ready");

Max.addHandler("set", (k, v) => {
  if (!(k in P)) { Max.outlet(["error", `unknown key: ${k}`]); return; }
  if (typeof P[k] === "number") P[k] = Number(v);
  else P[k] = String(v);
  Max.outlet(["param", k, P[k]]);
});

function sanitizeName(name, fallbackBase){
  let n = String(name||"").trim();
  if (!n) n = String(fallbackBase||"ir_mono48k");
  n = n.replace(/\.[^/.]+$/,"");                // 확장자 제거
  n = n.replace(/[^A-Za-z0-9_\-가-힣 ]+/g,"_");  // 안전 문자만(한글/공백 허용)
  return n + ".wav";
}

function resolveOutPath(outLoc, nameArg, inPath){
  const baseFromIn = path.basename(inPath).replace(/\.[^/.]+$/,"") + "_mono48k";
  const finalName  = sanitizeName(nameArg, baseFromIn);

  if (fs.existsSync(outLoc) && fs.statSync(outLoc).isDirectory()){
    // 폴더 → 파일명 붙이기
    return path.join(outLoc, finalName);
  }
  // 파일 경로 → 같은 폴더에 name으로 강제 치환
  const dir = path.dirname(outLoc);
  if (!fs.existsSync(dir)) return null;
  return path.join(dir, finalName);
}

Max.addHandler("run", (py, script, outLoc, inIR, nameArg) => {
  // 1) 인자 검증
  if ([py,script,outLoc,inIR].some(bad) || ![py,script,outLoc,inIR].every(isAbs)){
    Max.outlet(["error",
      `invalid paths\npy:${py}\nsc:${script}\nout:${outLoc}\nin:${inIR}\n(name ok to be empty)`]);
    return;
  }

  // 2) 출력 경로 결정 (이름 우선)
  let outPath = resolveOutPath(String(outLoc), String(nameArg||""), String(inIR));
  if (!outPath){ Max.outlet(["error", `Invalid output location: ${outLoc}`]); return; }
  if (!/\.(wav|wave)$/i.test(outPath)) outPath += ".wav";

  // 3) 파라미터 구성
  const args = [
    String(script),
    "--in",  String(inIR),
    "--out", String(outPath),
    "--mode", String(P.mode),
    "--early_ms", String(P.early_ms),
    "--late_start_ms", String(P.late_start_ms),
    "--xfade_ms", String(P.xfade_ms),
    "--target_sr", String(P.target_sr),
    "--peak_limit", String(P.peak_limit)
  ];
  if (P.mode === "ch") { args.push("--ch", String(P.ch)); }
  if (P.mode === "weighted" && P.weights && String(P.weights).trim()) {
    args.push("--weights", String(P.weights));
  }

  Max.outlet(["runargs", String(py), String(script), String(inIR), String(outPath)]);

  // 4) 실행
  const child = spawn(String(py), args, { shell:false });
  let out = "", err = "";
  child.stdout.on("data", d => out += d.toString());
  child.stderr.on("data", d => err += d.toString());
  child.on("close", code => {
    const text = (out + "\n" + err).trim();
    if (code !== 0 || /^usage:/mi.test(text)) {
      Max.outlet(["error", text || `python exited ${code}`]); return;
    }
    try {
      const j = JSON.parse(text);
      if (j && j.error) { Max.outlet(["error", text]); return; }
      Max.outlet(["done", outPath]);  // 최종 파일
      Max.outlet(["json", text]);
    } catch(e){
      Max.outlet(["error", "JSON parse failed\n"+text]);
    }
  });
});