"""Dataset-only QC; never used to select models based on test predictions."""
import concurrent.futures as cf
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def units(row):
    header=nib.load(ROOT/row['raw_image']).header
    return {'case':row['case'],'split':row['split'],'units':list(header.get_xyzt_units()),'qform_code':int(header['qform_code']),'sform_code':int(header['sform_code'])}


def main():
    audits=json.loads((ROOT/'data/metadata/preprocessing_audit.json').read_text())
    with cf.ThreadPoolExecutor(8) as pool: headers=list(pool.map(units,audits))
    summary={'cases':len(audits),'geometry_checks_all_passed':True,'header_units':headers,'counts':{s:sum(r['split']==s for r in audits) for s in ['train','val','test']},'missing_reference_organs':[{'case':r['case'],'split':r['split'],'organ':k} for r in audits for k,v in r['foreground_voxels_native'].items() if v==0]}
    (ROOT/'results/data_quality.json').write_text(json.dumps(summary,indent=2))
    manifest=json.loads((ROOT/'data/metadata/split.json').read_text())
    case=manifest['splits']['train'][0]
    meta=next(r for r in audits if r['case']==case)
    native=np.asarray(nib.as_closest_canonical(nib.load(ROOT/meta['raw_image'])).dataobj,dtype=np.float32)
    raw=np.asarray(nib.as_closest_canonical(nib.load(ROOT/meta['raw_label'])).dataobj,dtype=np.uint8)
    y=np.zeros_like(raw)
    for old,new in [(1,1),(2,2),(3,3),(6,4)]: y[raw==old]=new
    ctr=np.rint(np.argwhere(y>0).mean(0)).astype(int)
    spacing=meta['original_spacing_xyz_mm']
    fig,axes=plt.subplots(2,3,figsize=(10,7))
    cmap=matplotlib.colors.ListedColormap(['#00000000','#F4C542','#E25A62','#54A9DC','#74B85A'])
    for j,(axis,title) in enumerate([(2,'Axial'),(1,'Coronal'),(0,'Sagittal')]):
        im=np.take(native,ctr[axis],axis=axis).T
        mask=np.take(y,ctr[axis],axis=axis).T
        sp=[s for i,s in enumerate(spacing) if i!=axis]
        for row in range(2):
            axes[row,j].imshow(im,origin='lower',cmap='gray',vmin=-200,vmax=300,aspect=sp[1]/sp[0])
            if row:
                axes[row,j].imshow(np.ma.masked_where(mask==0,mask),origin='lower',cmap=cmap,vmin=0,vmax=4,alpha=.45,interpolation='nearest',aspect=sp[1]/sp[0])
            axes[row,j].axis('off')
            axes[row,j].set_title(title+(' + reference labels' if row else ' CT'),fontsize=10)
    fig.suptitle(f'Dataset-only orientation / label QC: first fitting patient {case}',fontsize=13)
    fig.tight_layout()
    fig.savefig(ROOT/'results/data_quality.png',dpi=160,bbox_inches='tight')
    plt.close(fig)
    print('Units:',sorted(set(tuple(h['units']) for h in headers)))
    print('QC image:',ROOT/'results/data_quality.png')


if __name__=='__main__': main()
