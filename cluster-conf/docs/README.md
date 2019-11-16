- Multi-label Semantic Segmentation with TF2
- Hyper-parameter tunning with Ray
- Why reproducibility is imp
- e2e reproducible pipeline on k8s
- Model release
- Gitops
- Model tracing monitoring 


!pygmentize 


          export MODEL_WEIGHTS="./pfs/evaluate/model.h5"
          export CALIB_WEIGHTS="./pfs/evaluate/calibration.weights"

          export MODEL_DB="evaluate"
          export MODEL_VERSION="c517ec50c54d4b83917972dacf7045d7"


           import tensorflow as tf
           import seldon_core
           from seldon_core.seldon_client import SeldonClient
           
           img_fn = "tf-data/test/Abyssinian_1/image.jpg"
           image = tf.io.read_file(img_fn)
           image = tf.image.decode_jpeg(image, channels=3)
           data = image.numpy()
          
           
           c = SeldonClient(deployment_name='petset-c517ec',namespace='kubeflow', gateway_endpoint='localhost:8080')
           r = sc.predict(gateway="istio", transport="rest",shape=(128, 128, 3), data=data, payload_type='ndarray', names=[])
           predictions = seldon_core.utils.seldon_message_to_json(r.response)
         
    
    
> https://istio.io/docs/tasks/telemetry/metrics/using-istio-dashboard/
     
kubectl -n istio-system port-forward $(kubectl -n istio-system get pod -l app=grafana -o jsonpath='{.items[0].metadata.name}') 3000:3000 &
  
http://localhost:3000/dashboard/db/istio-mesh-dashboard i
           