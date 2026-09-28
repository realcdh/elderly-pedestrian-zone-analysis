# -*- coding: utf-8 -*-
"""공공데이터포털 표준데이터 CSV 다운로더 (무인증, 세션 쿠키만 사용)"""
import re, sys, urllib.parse, subprocess, os, json

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0 Safari/537.36"
CJ = "cj.txt"

def curl(url, data=None, extra=None, out=None, timeout=300):
    cmd = ["curl","-sL","--max-time",str(timeout),"-A",UA,"-b",CJ,"-c",CJ,
           "-H","Referer: https://www.data.go.kr/"]
    if extra: cmd += extra
    if data is not None:
        cmd += ["--data", data]
    if out: cmd += ["-o", out, "-w", "%{http_code} %{size_download}"]
    cmd.append(url)
    r = subprocess.run(cmd, capture_output=True)
    return r.stdout if not out else r.stdout.decode()

def fetch_std(pk, dest):
    page = curl(f"https://www.data.go.kr/data/{pk}/standard.do").decode("utf-8","ignore")
    title = re.search(r"var publicDataSj\s*=\s*'([^']+)'|publicDataSj = '([^']+)'", page)
    t = re.search(r"<title>([^<|]+)", page)
    name = (t.group(1).strip() if t else f"pk{pk}")
    lst = curl(f"https://www.data.go.kr/tcs/dss/stdFileList.do?publicDataPk={pk}",
               extra=["-H","X-Requested-With: XMLHttpRequest"]).decode("utf-8","ignore")
    dpk = re.search(r'data-public-pk="([^"]+)"', lst)
    if not dpk: return name, None, "detailPk 없음"
    detail = curl("https://www.data.go.kr/tcs/dss/selectDpkDetailInfo.do",
                  data="publicDataDetailPk=" + urllib.parse.quote(dpk.group(1)),
                  extra=["-H","X-Requested-With: XMLHttpRequest",
                         "-H","Content-Type: application/x-www-form-urlencoded; charset=UTF-8"]
                  ).decode("utf-8","ignore")
    m = re.search(r"fn_fileDataDown\('(\d+)',\s*'([^']+)',\s*'([^']+)',\s*'([^']+)',\s*'([^']+)'\)", detail)
    if not m: return name, None, "fn_fileDataDown 없음"
    _, dpk2, atch, sn, ext = m.groups()
    q = urllib.parse.urlencode({"atchFileId":atch,"fileDetailSn":sn,"dataNm":name,
                                "publicDataPk":pk,"publicDataDetailPk":dpk2,"fileExtsn":ext})
    res = curl("https://www.data.go.kr/cmm/cmm/fileDownload.do?"+q, out=dest, timeout=900)
    return name, dest, res

if __name__ == "__main__":
    for pk, dest in json.loads(sys.argv[1]):
        print(pk, fetch_std(pk, dest), flush=True)
