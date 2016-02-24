### 2016 W Labor PS1                         ###
### by Hyunmin Park                          ###    
### based on example code by Thibaut Lamadon ###

### Import packages
require(gtools)
require(data.table)
require(ggplot2)
require(reshape)
require(SparseM)

### Useful functions
lognormpdf <- function(Y,mu=0,sigma=1){
  -0.5*((Y-mu)/sigma)^2 -0.5*log(2.0*pi)-log(sigma)
}

logsumexp <- function(v){
  vm = max(v)
  log(sum(exp(v-vm)))+vm
}

### Model simulation
model.mixture.new <-function(nk) {
  
  model = list()
  # model for Y1,Y2,Y3|k 
  model$A     = array(3*(1 + 0.8*runif(3*nk)),c(3,nk))
  model$S     = array(1,c(3,nk))
  model$pk    = rdirichlet(1,rep(1,nk))
  model$nk    = nk
  return(model)
}

model.mixture.simulate <-function(model,N,sd.scale=1) {
  
  Y1 = array(0,sum(N)) 
  Y2 = array(0,sum(N)) 
  Y3 = array(0,sum(N)) 
  K  = array(0,sum(N)) 
  
  A   = model$A
  S   = model$S
  pk  = model$pk
  nk  = model$nk
  
  # draw K
  K = sample.int(nk,N,TRUE,pk)
  
  # draw Y1, Y2, Y3
  Y1  = A[1,K] + S[1,K] * rnorm(N) *sd.scale
  Y2  = A[2,K] + S[2,K] * rnorm(N) *sd.scale
  Y3  = A[3,K] + S[3,K] * rnorm(N) *sd.scale
  
  data.sim = data.table(k=K,y1=Y1,y2=Y2,y3=Y3)
  return(data.sim)  
}

set.seed(17)
model = model.mixture.new(3)
data = model.mixture.simulate(model,10000,sd.scale=0.5) # simulating with lower sd to see separation
datal = melt(data,id="k")
ggplot(datal,aes(x=value,group=k,fill=factor(k))) + geom_density() + facet_grid(~variable) + theme_bw()

### Estimation
Niter=1000

# Initializing variables
Y1=data$y1
Y2=data$y2
Y3=data$y3
N=length(Y1)
nk=length(unique(data$k))
Dkj    = as.matrix.csr(kronecker(rep(1,N),diag(nk)))
DY1     = as.matrix(kronecker(Y1 ,rep(1,nk)))
DY2     = as.matrix(kronecker(Y2 ,rep(1,nk)))
DY3     = as.matrix(kronecker(Y3 ,rep(1,nk)))
A=array(0,c(3,nk))
S=array(2,c(3,nk))
set.seed(7)
pk = rdirichlet(1,rep(1,nk))
pkInit=pk
tau = array(0,c(N,nk))
lpm = array(0,c(N,nk))
lik = 0
likMat = array(0,Niter)
tauMat = array(0,c(N,nk,Niter))
lpmMat = array(0,c(N,nk,Niter))
for (iter in 1:Niter){
  for (i in 1:N) {
    ltau = log(pk)
    lnorm1 = lognormpdf(Y1[i], A[1,], S[1,])
    lnorm2 = lognormpdf(Y2[i], A[2,], S[2,])
    lnorm3 = lognormpdf(Y3[i], A[3,], S[3,])
    lall = ltau + lnorm2 + lnorm1 +lnorm3
    lpm[i,] = lall
    lik = lik + logsumexp(lall)
    tau[i,] = exp(lall - logsumexp(lall))
  }
  likMat[iter] = lik
  tauMat[,,iter] = tau
  lpmMat[,,iter] = lpm
  # Updating the A and S matrices
  rw     = c(t(tau))
  fit1   = slm.wfit(Dkj,DY1,rw)
  fit2   = slm.wfit(Dkj,DY2,rw)
  fit3   = slm.wfit(Dkj,DY3,rw)
  A[1,]  = coef(fit1)[1:nk]
  A[2,]  = coef(fit2)[1:nk]
  A[3,]  = coef(fit3)[1:nk]
  res1   = DY1-Dkj%*%A[1,] 
  res2   = DY2-Dkj%*%A[2,] 
  res3   = DY3-Dkj%*%A[3,] 
  fitv1  = slm.wfit(Dkj,res1^2,rw)
  fitv2  = slm.wfit(Dkj,res2^2,rw)
  fitv3  = slm.wfit(Dkj,res3^2,rw)
  S[1,] = sqrt(coef(fitv1)[1:nk])
  S[2,] = sqrt(coef(fitv2)[1:nk])
  S[3,] = sqrt(coef(fitv3)[1:nk])
  # Updating pk
  pk = colMeans(tau)
}

# Computing the Q and H

