import copy
import unittest
from score_body import score, reference_scales


def truth(index=0, video="video-a", scale=1):
    coords=[(.4,.2),(.6,.2),(.3,.3),(.7,.3),(.2,.4),(.8,.4),(.4,.5),(.6,.5),(.4,.7),(.6,.7),(.4,.9),(.6,.9)]
    return {"videoSHA256":video,"frameIndex":index,"session":"s","status":"reviewed","width":1000*scale,"height":1000*scale,
            "reviewMethod":"frame","points":[{"x":x,"y":y,"visibility":2,"source":"manual"} for x,y in coords]}


def prediction(row, offset=0):
    return {"videoSHA256":row["videoSHA256"],"frameIndex":row["frameIndex"],"detected":True,
            "prediction":[[p['x']*row['width']+offset,p['y']*row['height']] for p in row['points']],"keypointConfidences":[.9]*12}


class BodyScoreTests(unittest.TestCase):
    def test_reference_scale_and_pixel_mapping(self):
        r=truth()
        self.assertAlmostEqual(reference_scales([r])["video-a"],700)
        result=score([r],[prediction(r)])
        self.assertEqual(result['jointPCK0.05BodyScale'],1)
        self.assertEqual(result['wrists']['referenceJoints'],2)

    def test_missing_frame_and_low_confidence_remain_failures(self):
        a,b=truth(),truth(1)
        p=prediction(a)
        p['keypointConfidences'][4]=.2
        result=score([a,b],[p])
        self.assertEqual(result['referenceJoints'],24)
        self.assertEqual(result['within0.05BodyScale'],11)
        self.assertEqual(result['missingPredictionRecords'],1)
        self.assertEqual(result['missingJoints'],13)

    def test_scale_normalization_and_threshold_inclusive(self):
        a,b=truth(),truth(video='video-b',scale=2)
        self.assertEqual(score([a],[prediction(a,35)])['jointPCK0.05BodyScale'],1)
        self.assertEqual(score([b],[prediction(b,70)])['jointPCK0.05BodyScale'],1)
        self.assertEqual(score([a],[prediction(a,36)])['jointPCK0.05BodyScale'],0)

    def test_frame_review_does_not_erase_interpolation(self):
        r=truth(); r['points'][4]['source']='interpolated'
        result=score([r],[prediction(r)])
        self.assertEqual(result['frameReviewedSubset']['referenceJoints'],12)
        self.assertEqual(result['byPointSource']['interpolated']['referenceJoints'],1)
        self.assertEqual(result['noInterpolationFrameSubset']['referenceJoints'],0)
        self.assertEqual(result['frameReviewedNoInterpolationSubset']['referenceJoints'],0)

    def test_clean_legacy_frame_is_not_individually_reviewed(self):
        r=truth();del r['reviewMethod']
        result=score([r],[prediction(r)])
        self.assertEqual(result['noInterpolationFrameSubset']['referenceJoints'],12)
        self.assertEqual(result['frameReviewedNoInterpolationSubset']['referenceJoints'],0)

    def test_hidden_not_primary_visible_and_unlabeled_not_scored(self):
        r=truth();r['points'][4]['visibility']=1;r['points'][5]['visibility']=0
        result=score([r],[prediction(r)])
        self.assertEqual(result['referenceJoints'],10)
        self.assertEqual(result['hiddenEstimatedTargets']['referenceJoints'],1)
        self.assertEqual(result['wrists']['referenceJoints'],0)

    def test_macro_video_and_out_of_frame(self):
        rows=[truth(i) for i in range(3)]+[truth(video='video-b')]
        result=score(rows,[prediction(r) for r in rows[:3]])
        self.assertEqual(result['jointPCK0.05BodyScale'],.75)
        self.assertEqual(result['macroVideoJointPCK0.05BodyScale'],.5)
        p=prediction(rows[0]);p['prediction'][0]=[-1,200]
        self.assertEqual(score([rows[0]],[p])['missingJoints'],1)

    def test_duplicates_extra_predictions_and_wrong_joint_count_fail(self):
        r=truth();p=prediction(r)
        for rows,preds in (([r,r],[]),([r],[p,p]),([r],[prediction(truth(2))])):
            with self.assertRaises(ValueError):score(rows,preds)
        p=copy.deepcopy(p);p['prediction'].append([0,0])
        with self.assertRaises(ValueError):score([r],[p])


if __name__=='__main__':unittest.main()
