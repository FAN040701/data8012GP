"""Choose one new policy using development scores for the planned seed repeat."""
import json
from slice_selection import OUT, ROOT, CFG, run_dir


def main():
    scheme=json.loads((ROOT/'configs/slice_selection_repeat.json').read_text())
    val={}
    for name in scheme['candidates']:
        scores=[]
        for seed in CFG['seeds']:
            record=json.loads((run_dir(name+'_aug',seed)/'complete.json').read_text())
            scores.append(record['best_validation_macro_dice'])
        val[name]=sum(scores)/len(scores)
    selected=max(val,key=val.get)
    data=dict(selected_policy=selected,mean_augmented_development_dice=val,
              selection_rule=scheme['rule'],seeds=CFG['seeds'])
    (OUT/'selection.json').write_text(json.dumps(data,indent=2))
    print(selected)


if __name__=='__main__':
    main()
