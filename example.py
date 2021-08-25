from tensorflow.keras.preprocessing import image
from kantan_ml import KantanML, KantanModel
import argparse
import glob, os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'inference'], help='mode: train or inference')
    parser.add_argument('--model', type=str, default='mini_cnn', help='enter your model (mini_dnn, mini_cnn, efficientnetb0)')
    parser.add_argument('--image-num', type=int, default=5, help='enter image number')
    parser.add_argument('--labels', type=str, nargs='+', help='enter your labels')
    parser.add_argument('--save-dir', type=str, help='enter your select directory')
    parser.add_argument('--images-url', type=str, nargs='+', help='enter your inference image')

    args = parser.parse_args()

    if args.mode == 'train':
        ml = KantanML('http://localhost:4444/wd/hub', 
                      output_dir=os.path.dirname(args.save_dir),
                      output_file=os.path.basename(args.save_dir))
        ml.download_images(args.labels, args.image_num)
        model = ml.get_model(args.model)
        model = ml.train(model)
    
    if args.mode == 'inference':
        labels = args.labels
        model_path = sorted(glob.glob(os.path.join(args.save_dir, 'models/*.h5')))[-1]
        kmodel = KantanModel(model_path, labels)
        for img_url in args.images_url:
            outputs = kmodel.inference_for_url_probability(img_url)
            outputs = sorted([ [k, outputs[k]]for k in outputs.keys() ], key=lambda x:x[1], reverse=True)
            print()
            print(' - - - - inference image - - - - ')
            print('URL:', img_url)
            print('INFERRED LABELS:')
            for i, (k, p) in enumerate(outputs): 
                print(f' [ {i+1} ] { k }: { round(p*100, 2) }%')
            print(' - - - - - - - - - - - - - - - - ')