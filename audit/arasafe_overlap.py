# Run 2026-07-29 (UTC). Prompt overlap between AraSafe and the ArGuard Task B CSVs. Printed: exact overlap 0 in every split; normalized overlap 4 (train), 0 (dev).
# Body unchanged from the logged run; it reads the public files directly.
import urllib.request, json, csv, io, re
def get(u): return urllib.request.urlopen(u).read().decode('utf-8')
nat=[json.loads(l) for l in get("https://raw.githubusercontent.com/qcri/AraSafe-benchmark/main/human_written/arasafe_natural_prompts.jsonl").splitlines() if l.strip()]
syn=[json.loads(l) for l in get("https://raw.githubusercontent.com/qcri/AraSafe-benchmark/main/synthetic/arasafe_synthetic_prompts.jsonl").splitlines() if l.strip()]
ara = [(r.get('User_Prompt') or r.get('Prompt'), r['Safety_Label'], 'nat') for r in nat] + \
      [(r.get('User_Prompt') or r.get('Prompt'), r['Safety_Label'], 'syn') for r in syn]

base="https://raw.githubusercontent.com/araieval/ArGuard-2026-tasks/main/taskB/data/"
def load(f):
    return list(csv.DictReader(io.StringIO(get(base+f))))
tr=load("train.csv"); dv=load("dev_with_label.csv"); dvn=load("dev_without_label.csv")
print("rows train/dev/devnolabel:", len(tr), len(dv), len(dvn))
print("train cols:", tr[0].keys())

def norm(s):
    s = re.sub(r'[ؗ-ًؚ-ْٰـ]', '', s or '')  # diacritics+tatweel
    s = s.replace('أ','ا').replace('إ','ا').replace('آ','ا').replace('ة','ه').replace('ى','ي').replace('ؤ','و').replace('ئ','ي')
    s = re.sub(r'[^\w؀-ۿ]+',' ', s)
    return re.sub(r'\s+',' ',s).strip()

ara_exact = {p.strip() for p,_,_ in ara}
ara_norm  = {}
for p,l,src in ara: ara_norm.setdefault(norm(p), (l,src))

for name, rows in [("train",tr),("dev_with_label",dv),("dev_without_label",dvn)]:
    ex = sum(1 for r in rows if r['prompt'].strip() in ara_exact)
    nm = sum(1 for r in rows if norm(r['prompt']) in ara_norm)
    print(f"{name}: n={len(rows)} exact-overlap-with-AraSafe={ex} normalized-overlap={nm}")
