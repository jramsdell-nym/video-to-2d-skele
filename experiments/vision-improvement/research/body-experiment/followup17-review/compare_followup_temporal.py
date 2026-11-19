"""Read saved outputs from two experiments; reuse the reviewed temporal metric."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
FIRST = ROOT.parent/'session-holdout-20261111'
FOLLOWUP = ROOT.parents[2]/'training/body-conservative-20261111'
sys.path.insert(0,str(ROOT.parent))
import analyze_body_temporal


if __name__ == '__main__':
    paths = {'generic':FIRST/'generic-val-predictions.json',
             'candidate':FOLLOWUP/'candidate-val-predictions.json',
             'macVision':FIRST/'mac-vision-cpu/predictions.json'}
    truth = FIRST/'val-rows.json'
    rows = json.loads(truth.read_text())
    assert rows == json.loads((FOLLOWUP/'val-rows.json').read_text())
    result = analyze_body_temporal.analyze(rows,{k:json.loads(p.read_text()) for k,p in paths.items()})
    sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    result['inputs'] = {'truthSHA256':sha(truth),'wrapperSHA256':sha(Path(__file__)),
                        'metricSourceSHA256':sha(Path(analyze_body_temporal.__file__)),
                        'predictionsSHA256':{k:sha(p) for k,p in paths.items()},
                        'candidateMeaning':'20-epoch conservative17-slot follow-up, mapped to body slots5:17'}
    output = ROOT/'body-temporal-comparison.json'
    if output.exists():
        raise ValueError('Refusing to overwrite completed follow-up comparison')
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(output),'eligibleJointPairs':result['eligibleJointPairs'],
                      'commonObservedJointPairs':result['commonObservedJointPairs']}))
