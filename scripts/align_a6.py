#!/usr/bin/env python3.8
"""A6: class-wise image-text alignment for DFN-CLIP frozen features.
Measures whether the frozen image features contain the class signal (high
correct-class similarity) but the text prototype is often NOT the top-1
(correct class a close but non-winning competitor) => text-image alignment
bottleneck, not representation failure. Uses the EN-Photo base template
(the zero-shot decision rule). No new data; reuses cached test features.
"""
import numpy as np, torch, open_clip, json, os
CKPT='checkpoints/dfn_clip_vitl14/open_clip_pytorch_model.bin'
NAMES=['anthracnose','black_shank','brown_spot','cmv','frog_eye','genetic_abnormality',
       'healthy','nematodes','potato_tuber_moth','pvy','sunscald','target_spot','tmv',
       'tswv','weather_fleck','wildfire']  # index order (alphabetical, DatasetFolder default)
Tmpl="a photo of a tobacco leaf with {name} disease"
device='cuda' if torch.cuda.is_available() else 'cpu'
m,_,_=open_clip.create_model_and_transforms('ViT-L-14',pretrained=CKPT)
m.to(device).eval()
toks=open_clip.tokenize([Tmpl.format(name=n) for n in NAMES])
with torch.no_grad():
    te=m.encode_text(toks.to(device)); te=te/te.norm(dim=1,keepdim=True)
    te=te.cpu().numpy()
z=np.load('data/clip_features_clean.npz')
im=z['test_feats'].astype(np.float32); im=im/(np.linalg.norm(im,axis=1,keepdims=True)+1e-9)
y=z['test_labels']
S=im@te.T  # (138,16) cosine
n=len(y); correct=S[np.arange(n),y]
maxinc=np.max(np.where(np.arange(16)[None,:]==y[:,None],-1e9,S),axis=1)
margin=correct-maxinc
order=np.argsort(-S,axis=1)
rank=np.array([list(order[i]).index(int(y[i]))+1 for i in range(n)])
res={'mean_correct_sim':round(float(correct.mean()),4),'mean_max_incorrect_sim':round(float(maxinc.mean()),4),
     'mean_margin':round(float(margin.mean()),4),'median_margin':round(float(np.median(margin)),4),
     'rank_top1':round(float((rank==1).mean()),4),'rank_top3':round(float((rank<=3).mean()),4),
     'rank_top5':round(float((rank<=5).mean()),4),'rank_top10':round(float((rank<=10).mean()),4),
     'mean_rank':round(float(rank.mean()),2),
     'class_wise':[{'name':NAMES[j],'n':int((y==j).sum()),'mean_correct_sim':round(float(S[y==j,j].mean()),4) if (y==j).any() else None,'mean_rank':round(float(rank[y==j].mean()),2) if (y==j).any() else None} for j in range(16)]}
print(json.dumps(res,indent=1,ensure_ascii=False))
json.dump(res,open('data/align_a6.json','w'),ensure_ascii=False)
print("\nSaved data/align_a6.json")
