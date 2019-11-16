#!/usr/bin/env python
_template = '''
---
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: {app}
  namespace: argocd
spec:
  project: default
  syncPolicy:
    automated:
      prune: true
      selfHeal: true   
  source:
    repoURL: https://github.com/suneeta-mall/e2e-ml-on-k8s.git
    targetRevision: HEAD
    path: {path}
  destination:
    server: https://kubernetes.default.svc
    namespace: kubeflow
'''

kf_apps = ['api-service', 'application', 'application-crds', 'argo', 'bootstrap', 'centraldashboard', 'istio',
           'istio-crds', 'istio-install', 'jupyter-web-app', 'katib-controller', 'katib-db', 'katib-manager',
           'katib-ui', 'metacontroller', 'metadata', 'metrics-collector', 'minio', 'mysql', 'notebook-controller',
           'persistent-agent', 'pipelines-runner', 'pipelines-ui', 'pipelines-viewer', 'profiles', 'pytorch-job-crds',
           'pytorch-operator', 'scheduledworkflow', 'seldon-core-operator', 'spartakus', 'suggestion', 'tensorboard',
           'tf-job-operator', 'webhook']

with open("e2e-ml-argocd-app.yaml", "w") as f:
    f.write(_template.format(app="namespace", path="cluster-conf/namespace"))
    for app in kf_apps:
        f.write(_template.format(app=app, path=f"cluster-conf/kubeflow/kustomize/{app}"))
    # Pachyderm configurations
    f.write(_template.format(app="pachyderm", path="cluster-conf/pachyderm"))
    f.write(_template.format(app="ml-user", path="cluster-conf/ml-user"))
