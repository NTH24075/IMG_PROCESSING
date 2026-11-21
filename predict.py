"""Predict one new image without any ground-truth input."""
import argparse
import cv2
from object_counting.io import ROOT, load_params, load_scene, detect_scene, read_image, resolve, new_run, digest
from object_counting.pipeline import count_image
from object_counting.visualization import save_prediction
from object_counting.experiments import touches_test, check_freeze, log_test_run


def predict(image_path, scene_id='auto', config='configs/final.toml'):
    """scene_id='auto' picks the scene by background matching. Returns (prediction, path, scene_id)."""
    image_path=resolve(image_path)
    # A renamed copy of a test image is still a test image; same freeze rule as run_eval.
    if image_path.is_file() and touches_test([dict(image_id=image_path.stem,sha256=digest(image_path))]):
        log_test_run(ROOT,event='single_image_prediction',image=str(image_path),config=str(config),**check_freeze(config))
    params, _ = load_params(config)
    image = read_image(image_path)
    if scene_id == 'auto':
        scene_id, _ = detect_scene(image)
    scene, background, roi = load_scene(scene_id)
    if image.shape[:2] != background.shape[:2]:
        # shortcut: plain resize assumes same camera framing; a cropped frame will misalign with the background.
        image = cv2.resize(image, background.shape[1::-1], interpolation=cv2.INTER_AREA)
    if scene.get('background_status') not in ('reviewed','legacy_offline_median'):
        print('Lưu ý: nền cảnh này còn cần nhóm rà soát.')
    prediction = count_image(image, background, params, roi, scene.get('perspective'))
    path = new_run('predictions')
    save_prediction(path, image, prediction, dict(image=str(image_path), scene_id=scene_id,
                                                background_status=scene.get('background_status')))
    return prediction, path, scene_id


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',required=True); parser.add_argument('--scene',default='auto',help="scene id hoặc 'auto' (tự nhận theo nền)")
    parser.add_argument('--config',default='configs/final.toml')
    args=parser.parse_args()
    try:
        result,path,scene_id=predict(args.image,args.scene,args.config)
    except (ValueError,OSError,KeyError) as error:
        parser.exit(2,str(error)+'\n')
    print(f'Cảnh: {scene_id} | Dự đoán: {result.count} | Kết quả: {path}')


if __name__=='__main__':
    main()
