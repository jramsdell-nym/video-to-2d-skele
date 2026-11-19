import copy
import unittest
from analyze_body_temporal import analyze


def truth(index, pts=None, shift=0):
    pts=index*10 if pts is None else pts
    return {'videoSHA256':'a','frameIndex':index,'status':'reviewed','width':1000,'height':1000,
            'frame':{'index':index,'pts':pts,'timeBase':'1/1000','time':pts/1000},
            'points':[{'x':.3+shift,'y':.3,'visibility':2,'source':'manual'} for _ in range(12)]}


def prediction(row, offset=0, detected=True):
    return {'videoSHA256':row['videoSHA256'],'frameIndex':row['frameIndex'],'detected':detected,
            'prediction':[[p['x']*1000+offset,p['y']*1000] for p in row['points']],
            'keypointConfidences':[.9]*12}


class BodyTemporalTests(unittest.TestCase):
    def test_correct_fast_motion_and_constant_offset_cancel(self):
        rows=[truth(0),truth(1,shift=.5)]
        out=analyze(rows,{'a':[prediction(r,10) for r in rows]})
        self.assertEqual(out['commonObservedJointPairs'],12)
        self.assertEqual(out['commonSupport']['allModels']['scores']['a']['allJoints']['residualDeltaPixels']['maximum'],0)

    def test_exact_irregular_timing_controls_rate(self):
        rows=[truth(0,0),truth(1,10),truth(2,40)]
        out=analyze(rows,{'a':[prediction(r,d) for r,d in zip(rows,[0,1,4])]})
        self.assertEqual(out['perModelAvailableSupport']['a']['allJoints']['pixelsPerPresentationSecond']['maximum'],100)

    def test_gaps_invalid_timing_and_nonpositive_time_break_pairs(self):
        for rows in ([truth(0),truth(2)],[truth(0),truth(1,0)]):
            out=analyze(rows,{'a':[prediction(r) for r in rows]})
            self.assertEqual(out['eligibleJointPairs'],0)
            self.assertEqual(len(out['breaks']),1)
        rows=[truth(0),truth(1)];rows[1]['frame']['time']=50
        out=analyze(rows,{'a':[prediction(r) for r in rows]})
        self.assertEqual(out['invalidTimingFrames'],1)
        self.assertEqual(out['eligibleJointPairs'],0)

    def test_common_support_and_misses_do_not_reward_selective_model(self):
        rows=[truth(0),truth(1),truth(2)]
        full=[prediction(r,d) for r,d in zip(rows,[0,50,0])]
        sparse=[prediction(rows[0]),prediction(rows[1],detected=False),prediction(rows[2])]
        out=analyze(rows,{'generic':full,'candidate':full,'macVision':sparse})
        self.assertEqual(out['eligibleJointPairs'],24)
        self.assertEqual(out['commonSupport']['allModels']['jointPairs'],0)
        self.assertEqual(out['commonSupport']['yoloPair']['jointPairs'],24)
        self.assertEqual(out['missingness']['macVision']['missingJoints'],12)
        self.assertEqual(out['perModelAvailableSupport']['macVision']['allJoints']['measuredJointPairs'],0)

    def test_joint_gates_and_reference_visibility_are_explicit(self):
        rows=[truth(0),truth(1)];rows[1]['points'][0]['visibility']=1
        preds=[prediction(r) for r in rows]
        preds[1]['keypointConfidences'][4]=.29
        preds[1]['prediction'][5]=[-1,100]
        out=analyze(rows,{'a':preds})
        self.assertEqual(out['eligibleJointPairs'],11)
        self.assertEqual(out['commonObservedJointPairs'],9)
        self.assertEqual(out['missingness']['a']['missingWrists'],2)
        self.assertEqual(out['missingness']['a']['visibleReferenceJoints'],23)

    def test_interpolation_missing_records_and_no_mutation(self):
        rows=[truth(0),truth(1)];rows[1]['points'][4]['source']='interpolated'
        preds=[prediction(r) for r in rows];original=copy.deepcopy((rows,preds))
        out=analyze(rows,{'a':preds,'b':preds[:1]})
        self.assertEqual(out['missingness']['b']['missingPredictionRecords'],1)
        self.assertEqual(out['perModelAvailableSupport']['a']['wrists']['measuredPairsWithInterpolatedReference'],1)
        self.assertEqual((rows,preds),original)

    def test_duplicate_extra_and_malformed_detected_predictions_fail(self):
        rows=[truth(0)];pred=prediction(rows[0])
        for truth_rows,preds in ((rows*2,[pred]),(rows,[pred,pred]),(rows,[prediction(truth(2))])):
            with self.assertRaises(ValueError):analyze(truth_rows,{'a':preds})
        pred['prediction'].pop()
        with self.assertRaises(ValueError):analyze(rows,{'a':[pred]})


if __name__=='__main__':
    unittest.main()
