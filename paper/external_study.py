"""Frozen two-model external test on the Jiaxing (ThyUS2Path) archives.

Run from the repository root (see README, "External evaluation"). Defaults reproduce the reported study:
checkpoints runs/development/best.pt (MobileNetV3-Small) and runs/resnet18/best.pt (ResNet-18),
Grad-CAM, one index view per patient. Pass --mobile-run/--resnet-run to evaluate another seed and
--cam gradcampp for an alternative attribution method. Existing keys of results.json are unchanged;
`effect_size` (logit-unit contrasts and the random-patch null rate) is added.
"""
import argparse, hashlib, io, json, sys
from pathlib import Path
try:
    import libarchive
except ImportError as exc:  # needs the libarchive-c package and the libarchive system library
    raise SystemExit("Install libarchive-c (pip install libarchive-c) and the libarchive library to read the .rar archives.") from exc
import numpy as np
import pandas as pd
from PIL import Image
import torch
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

def md5(path):
    h=hashlib.md5()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['project','archives','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--mobile-run',type=Path,default=Path('runs/development'),help='MobileNet run folder, relative to --project.')
    p.add_argument('--resnet-run',type=Path,default=Path('runs/resnet18'),help='ResNet-18 run folder, relative to --project.')
    p.add_argument('--cam',choices=['gradcam','gradcampp'],default='gradcam')
    p.add_argument('--threads',type=int,default=3)
    a=p.parse_args();sys.path.insert(0,str(a.project.resolve()))
    from scans import fit_image
    from models import load_checkpoint, predict, metrics
    from explanations import gradcam, centered_prior, patch_scores
    out=a.out;out.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(a.threads)
    ref=a.project/a.mobile_run
    training_hashes=set(pd.concat([pd.read_csv(ref/f).sha256 for f in ['split_manifest.csv','excluded_views.csv']]))
    items=[]
    missing=[]; unmatched_files=[]
    for batch,checksum in enumerate(['a6fb578883ae70a08fd9e92fa7da7ab5','08d02473840c0e916d5edf6a462026d8'],1):
        archive=a.archives/f'batch{batch}.rar'
        assert md5(archive)==checksum,f'{archive} failed its published MD5 digest'
        folder=a.archives/f'extracted{batch}';folder.mkdir(exist_ok=True)
        with libarchive.file_reader(str(archive)) as entries:
            for e in entries:
                name=Path(str(e.pathname)).name
                if name.lower().endswith(('.jpg','.csv','.jpeg')):
                    dest=folder/name
                    if not dest.exists():dest.write_bytes(b''.join(e.get_blocks()))
        stem=f'batch{batch}_image'
        df=pd.read_csv(folder/(stem+'.csv')).merge(pd.read_csv(folder/(stem+'_label.csv')),on='patient_name',validate='many_to_one')
        index={f.name.casefold():f for f in folder.glob("*") if f.suffix.lower() in {".jpeg",".jpg"}}
        unmatched_files.extend((batch,n) for n in sorted(set(index)-set(df.path.str.casefold())))
        for row in df.itertuples():
            path=index.get(row.path.casefold())
            if path is None:
                missing.append((batch,row.patient_name,row.path));continue
            assert str(row.patient_name)==path.name.split("_")[0]
            with Image.open(path) as im:
                g=im.convert('L');key=hashlib.sha256(f'{g.width}x{g.height}:'.encode()+g.tobytes()).hexdigest()
            items.append(dict(batch=batch,patient=f'{batch}_{row.patient_name}',image_id=f'{batch}_{row.path}',image_path=str(path),label=row.histo_label,sha256=key))
    pd.DataFrame(missing,columns=["batch","patient","metadata_filename"]).to_csv(out/"unmatched_metadata.csv",index=False)
    pd.DataFrame(unmatched_files,columns=["batch","archive_filename"]).to_csv(out/"unmatched_files.csv",index=False)
    data=pd.DataFrame(items)
    assert data.groupby('patient').label.nunique().max()==1 and data.image_id.is_unique
    duplicate=data.sha256.duplicated(keep=False)|data.sha256.isin(training_hashes)
    data.loc[duplicate].drop(columns='image_path').to_csv(out/'excluded.csv',index=False)
    data=data[~duplicate].sort_values(['patient','image_id']).reset_index(drop=True)
    first=data.drop_duplicates('patient').copy()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    results={}
    for name,folder in [('mobile',a.mobile_run),('resnet',a.resnet_run)]:
        model,ck=load_checkpoint(a.project/folder/'best.pt',device)
        pred=predict(model,data,device).set_index('image_id')
        data[name]=data.image_id.map(pred.score)
        gaps=[];nulls=[]
        for r in first.itertuples():
            with Image.open(r.image_path) as image:gray,content=fit_image(image)
            logit,heat=gradcam(model,ck['architecture'],gray,device,a.cam)
            rr=patch_scores(model,gray,content,heat,centered_prior(content),logit,device)
            gaps.append(rr['gap']);nulls.append(rr['null_positive_rate'])
        first[name+'_gap']=gaps;first[name+'_null']=nulls
        results[name]={'threshold':ck['threshold']}
        print(name,'finished',flush=True)
    cases=data.groupby(['patient','batch','label'],as_index=False)[['mobile','resnet']].mean().merge(first[['patient','image_id','mobile_gap','resnet_gap','mobile_null','resnet_null']],on='patient',validate='one_to_one')
    assert cases[['mobile','resnet']].notna().all().all()
    for name in results:
        for group,frame in [('all',cases),*list(cases.groupby('batch'))]:
            results[name][str(group)]=metrics(frame.label,frame[name],results[name]['threshold'])
    m=cases.query('label==1').dropna(subset=['mobile_gap','resnet_gap'])
    results['hotspot']={'n':len(m),'mobile':int(m.mobile_gap.gt(0).sum()),'resnet':int(m.resnet_gap.gt(0).sum())}
    rng=np.random.default_rng(73);pos=cases.query('label==1')[['mobile','resnet']].to_numpy();neg=cases.query('label==0')[['mobile','resnet']].to_numpy()
    delta=m.mobile_gap.gt(0).astype(int)-m.resnet_gap.gt(0).astype(int)
    boots=[]
    for _ in range(2000):
        v=np.concatenate([rng.choice(pos,len(pos)),rng.choice(neg,len(neg))])
        auc=(rankdata(v,axis=0)[:len(pos)].sum(axis=0)-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
        boots.append([*auc,auc[1]-auc[0],rng.choice(delta,len(delta)).mean()])
    results['ci_columns']=['mobile_auc','resnet_auc','resnet_minus_mobile_auc','mobile_minus_resnet_hotspot_rate']
    results['ci']=np.quantile(boots,[.025,.975],axis=0).tolist()
    # Effect sizes in logit units. A separate generator keeps the reported bootstrap above bit-for-bit unchanged.
    eff=np.random.default_rng(74);d=(m.mobile_gap-m.resnet_gap).to_numpy()
    def ci(v):return np.quantile([eff.choice(v,len(v)).mean() for _ in range(2000)],[.025,.975]).tolist()
    results['effect_size']={n:{'mean_gap':float(m[n+'_gap'].mean()),'mean_gap_95ci':ci(m[n+'_gap'].to_numpy()),
        'mean_null_positive_rate':float(m[n+'_null'].mean()),'positive_rate':float(m[n+'_gap'].gt(0).mean())} for n in ('mobile','resnet')}
    results['effect_size']['mobile_minus_resnet_mean_gap']={'estimate':float(d.mean()),'95ci':ci(d)}
    results['cam']=a.cam
    results.update(matched_views=len(items),missing_view_links=len(missing),unlinked_files=len(unmatched_files),retained_views=len(data),retained_patients=len(cases),excluded=int(duplicate.sum()))
    cases.to_csv(out/'patients.csv',index=False);data.drop(columns='image_path').to_csv(out/'views.csv',index=False)
    (out/'results.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
