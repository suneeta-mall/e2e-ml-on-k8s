# Environment configuration for e2e-ml-on-k8s using gitops

This section caters to configuring environment and runtime for ML related operations which includes:
 1. Jupter notebook on k8s (powered by [Kubeflow](https://github.com/kubeflow/kubeflow))
 2. DAG based pipelining solution [Kubeflow](https://github.com/kubeflow/kubeflow), [Argo](https://github.com/argoproj/argo) and [Pachyderm](https://github.com/pachyderm/pachyderm)
 3. Training operators for Tensorflow, Pytorch and lots more (powered by [Kubeflow](https://github.com/kubeflow/kubeflow))
 4. Hyper parameter tuning [Katib](https://github.com/kubeflow/katib)
 5. Model serving tooklits (powered by [Seldon](https://github.com/SeldonIO/seldon-core/) & [Kubeflow](https://github.com/kubeflow/kubeflow))
 

Version of software has been used in this configuration files:
- Kubernetes: 1.14.7  (tested on this version, in theory should work with other versions too!)
- ArgoCD:     1.2.3
- Kubeflow:   0.6.2
- Seldon      0.4.1  (upgraded from packaged version on kubeflow 0.6.2)
- Pachyderm:  1.9.7


## Configuring Kubernetes cluster with gitops

GitOps is well explained [here](https://www.weave.works/technologies/gitops/) and [here](https://argoproj.github.io/argo-cd/). 
The configuration approached used here is based on gitops and is a three step process:
1. BYO Kubernetes cluster (for quick example see [this create kube cluster section](#create-kubernetes-cluster))
2. Install [ArgoCD](https://argoproj.github.io/argo-cd/) (details can be found in [this section](#cd-with-argocd))
3. Install ArgoApp for [e2e-ml-on-k8s](https://github.com/suneeta-mall/e2e-ml-on-k8s.git)
```bash
kubectl apply –f https://raw.githubusercontent.com/suneeta-mall/e2e-ml-on-k8s/master/cluster-conf/e2e-ml-argocd-app.yaml
```

Details on how the config files are generated is detailed [in this section](#generating-env-config-files). 
RBAC related to [ml-user](ml-user) is manually crafted as per need.

Borrowing the image from ArgoCD, GitOps pictorially:


![GitOps with ArgoCD](https://argoproj.github.io/argo-cd/assets/argocd_architecture.png "GitOps with ArgoCD")


## Connecting to cluster
Use kubectl and [pachctl](#port-forward-config) to connect to cluster and other services in cluster.

## Create Kubernetes cluster
### GKE
 ```bash
# make sure to use https://www.googleapis.com/auth/devstorage.read_write storage-rw
gcloud beta container --project "kubecon-cncf" clusters create "e2e-ml" --zone "us-east1-c" --no-enable-basic-auth --cluster-version "1.14.7-gke.14" --machine-type "n1-standard-1" --image-type "COS" --disk-type "pd-standard" --disk-size "100" --metadata disable-legacy-endpoints=true --scopes "https://www.googleapis.com/auth/devstorage.read_write","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" --num-nodes "1" --enable-stackdriver-kubernetes --enable-ip-alias --network "projects/kubecon-cncf/global/networks/default" --subnetwork "projects/kubecon-cncf/regions/us-east1/subnetworks/default" --default-max-pods-per-node "110" --addons HorizontalPodAutoscaling,HttpLoadBalancing --enable-autoupgrade --enable-autorepair && \
gcloud beta container --project "kubecon-cncf" node-pools create "pool-1" --cluster "e2e-ml" --zone "us-east1-c" --node-version "1.14.7-gke.14" --machine-type "n1-standard-4" --image-type "COS" --disk-type "pd-standard" --disk-size "100" --metadata disable-legacy-endpoints=true --scopes "https://www.googleapis.com/auth/devstorage.read_write","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" --enable-autoscaling --min-nodes "0" --max-nodes "8" --enable-autoupgrade --enable-autorepair && \
gcloud beta container --project "kubecon-cncf" node-pools create "pool-2" --cluster "e2e-ml" --zone "us-east1-c" --node-version "1.14.7-gke.14" --machine-type "n1-standard-4" --accelerator "type=nvidia-tesla-k80,count=1" --image-type "COS" --disk-type "pd-standard" --disk-size "100" --metadata disable-legacy-endpoints=true --scopes "https://www.googleapis.com/auth/devstorage.read_write","https://www.googleapis.com/auth/logging.write","https://www.googleapis.com/auth/monitoring","https://www.googleapis.com/auth/servicecontrol","https://www.googleapis.com/auth/service.management.readonly","https://www.googleapis.com/auth/trace.append" --enable-autoscaling --min-nodes "0" --max-nodes "3" --enable-autoupgrade --enable-autorepair
```

#### Configure GPU drivers and device plugins
```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

### AWS
Instructions to starting EKS cluster are [here](https://docs.aws.amazon.com/eks/latest/userguide/getting-started.html)
```bash
eksctl create cluster \
--name e2e-ml \
--version 1.14.7 \
--region us-east-1 \
--nodegroup-name pool-1 \
--node-type m4.medium \
--nodes 0 \
--nodes-min 0 \
--nodes-max 8 \
--node-ami auto
```

#### GPU driver setup:
```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/1.0.0-beta/nvidia-device-plugin.yml
```


## CD with ArgoCD
For details introduction on gitops based CD using ArgoCD see [here](https://argoproj.github.io/argo-cd/getting_started/).

```bash
# kubectl create clusterrolebinding suneetamall-cluster-admin-binding --clusterrole=cluster-admin --user=user@email.com
kubectl apply -n argocd -f cluster-conf/argo-cd.yaml
```

Useful port forward:
```bash
kubectl port-forward svc/argocd-server -n argocd 8080:443
```
### login ArgoCD
```bash
kubectl get pods -n argocd -l app.kubernetes.io/name=argocd-server -o name | cut -d'/' -f 2
argocd login localhost:8080
argocd account update-password
```

## Generating env config files
### Generate argo cd app
All the software required for reproducible ML platform are installed as argo app. Receipts for these are [kubeflow](kubeflow) and [pachyderm](pachyderm). Details of how these are generated in detailed in following sections.  
```bash
python generate_argo.py
```

#### Apply Argo Apps
```bash
kubectl apply -f cluster-conf/e2e-ml-argocd-app.yaml
```

### Generating kubeflow app
Kubeflow has been installed with Istio as per independent installation on existing Kubernetes. For details (see)[https://www.kubeflow.org/docs/started/k8s/kfctl-k8s-istio/]

```bash
export KFAPP=kubeflow
kfctl init ${KFAPP} --config=kubeflow/kfctl_k8s_istio.0.6.2.yaml -V
cd ${KFAPP}
kfctl generate all -V
kfctl apply all -V
```

Useful port forward:

```bash
kubectl port-forward -n istio-system svc/istio-ingressgateway 8080:80
#  http://localhost:3000/dashboard/db/istio-mesh-dashboard
kubectl -n istio-system port-forward $(kubectl -n istio-system get pod -l app=grafana -o jsonpath='{.items[0].metadata.name}') 3000:3000 &
```

#### Amends
Seldon operators were upgraded to 0.4.1 to include following operators:
```yaml
    - UNKNOWN_IMPLEMENTATION
    - SIMPLE_MODEL
    - SIMPLE_ROUTER
    - RANDOM_ABTEST
    - AVERAGE_COMBINER
    - SKLEARN_SERVER
    - XGBOOST_SERVER
    - TENSORFLOW_SERVER
    - MLFLOW_SERVER
```

                                    
### Generating Pachyderm
#### Backend bucket
```bash
gsutil mb -p kubecon-cncf gs://pach-e2e-ml
```
#### To generate pach deployment
```bash
export STORAGE_SIZE=50
export BUCKET_NAME=pach-e2e-ml
pachctl deploy google ${BUCKET_NAME} ${STORAGE_SIZE} --dynamic-etcd-nodes=1 --output yaml --dry-run \
     --namespace kubeflow > pachyderm.yaml
```
#### Port forward config
```bash
pachctl config update context `pachctl config get active-context` --namespace kubeflow
pachctl port-forward &  
```

### Configure CI/CD with gitops on e2e-k8s-on-ml
```bash
kubectl apply -f cluster-conf/e2e-ml-argocd-app.yaml
```
