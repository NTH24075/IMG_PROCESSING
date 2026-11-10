"""Predict one new image without any ground-truth input."""
import argparse
from object_counting.io import ROOT, load_params, load_scene, read_image, resolve, new_run
from object_counting.pipeline import count_image
from object_counting.visualization import save_prediction
from object_counting.evaluation_guard import authorize, audit


def predict(image_path, scene_id, config='configs/baseline.toml'):
    image_path=resolve(image_path)
    request=authorize([dict(image_id=image_path.stem,image=str(image_path),scene_id=scene_id,split='dev')],
                      resolve(config),'dev',purpose='single_image_prediction')
    params, _ = load_params(config)
    scene, background, roi = load_scene(scene_id)
    image = read_image(resolve(image_path))
    if scene.get('background_status') != 'reviewed':
        print('Lưu ý: nền cảnh này còn cần nhóm rà soát.')
    try:
        prediction = count_image(image, background, params, roi)
        path = new_run('predictions')
        save_prediction(path, image, prediction, dict(image=str(image_path), scene_id=scene_id,
                                                    background_status=scene.get('background_status')))
    except Exception as error:
        if request:
            audit(dict(request_id=request,status='failed',reason=str(error)))
        raise
    if request:
        audit(dict(request_id=request,status='completed',run_id=path.name))
    return prediction, path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',required=True); parser.add_argument('--scene',required=True)
    parser.add_argument('--config',default='configs/baseline.toml')
    args=parser.parse_args()
    try:
        result,path=predict(args.image,args.scene,args.config)
    except (ValueError,OSError,KeyError) as error:
        parser.exit(2,str(error)+'\n')
    print(f'Dự đoán: {result.count} | Kết quả: {path}')


if __name__=='__main__':
    main()
