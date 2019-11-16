See https://www.kubeflow.org/docs/other-guides/virtual-dev/getting-started-minikube/

```bash
export KUBEFLOW_TAG=v0.6.2

# curl -O https://raw.githubusercontent.com/kubeflow/kubeflow/${KUBEFLOW_TAG}/scripts/setup-minikube.sh      
 chmod +x setup-minikube.sh download.sh  
./setup-minikube.sh
```


```bash
minikube start --cpus 4 --memory 8096 --disk-size=40g --kubernetes-version='v1.14.6' 
```

```bash
minikube start --cpus 7 --memory 10096 --disk-size=40g --kubernetes-version='v1.14.6' 
pachctl deploy local
kfctl apply all -V

# kfctl delete all -V
# kubectl delete ns istio-system
# minikube delete
```

minikube tunnel

