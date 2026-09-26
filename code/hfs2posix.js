// hfs2posix.js  — HFS("Macintosh HD:Users:…") → POSIX("/Users/…") 변환
inlets = 1;
outlets = 1;

function anything() {
  var s = arrayfromargs(messagename, arguments).join(" ");
  outlet(0, toPosix(s));
}

function toPosix(p) {
  if (!p || typeof p !== "string") return "";
  // 이미 POSIX면 그대로
  if (p[0] === "/") return p;

  // HFS "Volume:Path:To:File" 매칭
  var m = p.match(/^([^:]+):(.*)$/);
  if (!m) return p;

  var vol = m[1];
  var rest = m[2].replace(/:/g, "/");
  if (rest.charAt(0) === "/") rest = rest.slice(1);

  // 부팅 볼륨(보통 "Macintosh HD")은 루트로 매핑
  if (vol === "Macintosh HD") return "/" + rest;

  // 외장/다른 볼륨
  return "/Volumes/" + vol + "/" + rest;
}