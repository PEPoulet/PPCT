library(adsim)
library(lattice)

data("posterior")
popPars <- apply(posterior, 2, median)

generateData <- function(nSubj, 
                         precision=NULL, 
                         placebo=FALSE, 
                         drugEffect=0.,
                         popParamsRandom=TRUE,
                         timePoints=seq(0, 52, 52),
                         seed=42) {

  ## Hyperparameters / tunable parameters
  set.seed(seed)
  
  ## Sampling population parameters for data generation according to simulation model
  if (popParamsRandom) {
    muEta <- rnorm(1, popPars['nuEta'], popPars['psiEta'])
    muAlpha <- rnorm(1, popPars['nuAlpha'], popPars['psiAlpha'])
    tauEta <- rgamma(1, popPars['kappaEta'], popPars['kappaEta'] * popPars['phiEta']^2)
    sigmaEta <- 1 / sqrt(tauEta)
    tauAlpha <- rgamma(1, popPars['kappaAlpha'], popPars['kappaAlpha'] * popPars['phiAlpha']^2)
    sigmaAlpha <- 1 / sqrt(tauAlpha)
  } else {
    muEta <- popPars['nuEta']
    muAlpha <- popPars['nuAlpha']
    sigmaEta <- popPars['phiEta']
    sigmaAlpha <- popPars['phiAlpha']
  }
  if (is.null(precision)) {
    tauResid <- rgamma(1, popPars['kappaEpsilon'], popPars['kappaEpsilon'] * popPars['phiEpsilon']^2)
  } else {
    tauResid <- precision
  }
  ## Inter-individual SD for "our study":
  #print(paste('Inter-individual SD for current simulation', sigmaEta * 70 / 4))
  ## Residual SD for "our study":
  #print(paste('Residual SD for current simulation', 70 * sqrt( (25/70) * (1 - 25/70) * 
  #                                                               (1/tauResid) / (1 + 1/tauResid))))
  
  ## Sampling baseline covariates
  bmmse <- runif(nSubj, 16, 30)
  ## Using tuned covariate simulator
  apo <- adsim:::.simApoE(nSubj, popPars)
  age <- adsim:::.simAge(nSubj, popPars, apo)
  gender <- adsim:::.simGen(nSubj, popPars)
  
  ## Computing individual parameters
  muEtaAdj <- muEta + popPars['lambdaEtaBMMSE'] * (bmmse - 21)
  muAlphaAdj <- muAlpha + popPars['lambdaAlphaBMMSE'] * (bmmse - 21) + 
    popPars['lambdaAlphaAge'] * (age -75) + popPars['lambdaAlphaApo1'] * 
    (apo==1) + popPars['lambdaAlphaApo2'] * (apo==2) + popPars['lambdaAlphaGen'] * gender
  
  ## Sample random effects
  eta <- rnorm(nSubj, muEtaAdj, sigmaEta)
  alpha <- rnorm(nSubj, muAlphaAdj, sigmaAlpha)
  
  ## Study time of visits
  times <- timePoints
  subjId <- 1:nSubj
  dat <- expand.grid(times, subjId)
  names(dat) <- c("Week", "SubjId")
  dat$BMMSE <- bmmse[dat$SubjId]
  dat$Eta <- eta[dat$SubjId] # (not available in a real data set)
  dat$Alpha <- alpha[dat$SubjId] # (not available in a real data set)
  #dat$LogitCondExp <- with(dat, Eta + Alpha * Week)
  
  ## Adding placebo effect
  kel <- popPars['kel']
  keq <- kel + popPars['keqMinusKel']
  if (placebo) {
    beta <- - popPars['aucPlacebo'] / (1 / kel - 1 / keq)
  } else {
    beta <- 0.
  }
  ePlacebo <- beta * ( exp( -kel * times) - exp( -keq * times ) )
  dat$LogitCondExp <- with(dat, Eta + Alpha * Week + ePlacebo)
  
  ## Adding symptomatic treatment effect
  #eStar <- popPars['eStar[1]']
  eStar <- drugEffect
  et50 <- popPars['et50[1]']
  gamma <- popPars['gamma[1]']
  b <- 12 / et50
  eDelta <- - (1 + b) * eStar / b
  eDon5 <- eDelta * times / (et50 + times)
  #dat$TreatmenEffect <- eDon5
  
  trts <- c("Placebo", "Treatment")
  dat$Treatment <- rep(trts, each = nSubj/2)[dat$SubjId]
  dat$Treatment <- factor(dat$Treatment, level = trts)
  dat <- within(dat, LogitCondExp[Treatment == "Treatment"] <-
                  LogitCondExp[Treatment == "Treatment"] + eDon5)
  
  dat$Theta <- with(dat, exp(LogitCondExp) / (1 + exp(LogitCondExp)))
  dat$Adas <- with(dat, 70 * rbeta(length(Theta), Theta * tauResid,
                                   (1 - Theta) * tauResid))
  return(dat)
}

dat <- generateData(nSubj=128, drugEffect = 0.)
write.csv(dat, 
          "/Users/p-e.poulet/Documents/Aramis/code/notebooks/PPI_CT/simulated_data/test.csv",
          row.names = FALSE,
          )

nbRepeats <- 8

## Exp 1 : Vary treatment effect size
trtEffect <- seq(0, 50, 1)
step <- 0.01
for (i in trtEffect) {
  trt <- i * step
  for (k in 1:nbRepeats) {
    data <- generateData(nSubj=200,
                         drugEffect = trt,
                         popParamsRandom = FALSE,
                         seed = i * 8 + k,
                         )
    write.csv(data, 
              paste('/Users/p-e.poulet/Documents/Aramis/code/notebooks/PPI_CT/simulated_data/Exp1/data_', trt, '_', k, '.csv',sep='',collapse=''),
              row.names = FALSE,
    )
  }
}

## Exp 2 : Vary outcome noise
nbRepeats <- 32
noises <- seq(0, 50, 1)
step <- .1
for (i in noises) {
  noise <- i * step
  for (k in 1:nbRepeats) {
    data <- generateData(nSubj=200,
                         drugEffect = popPars['eStar[1]'],
                         precision = noise,
                         popParamsRandom = FALSE,
                         seed = i * nbRepeats + k,
    )
    write.csv(data, 
              paste('/Users/p-e.poulet/Documents/Aramis/code/notebooks/PPI_CT/simulated_data/Exp2/data_', noise, '_', k, '.csv',sep='',collapse=''),
              row.names = FALSE,
    )
  }
}

## Exp 3 : Vary model noise
for (i in 0:50) {
  for (k in 1:nbRepeats) {
    data <- generateData(nSubj=200,
                         drugEffect = popPars['eStar[1]'],
                         precision = NULL,
                         popParamsRandom = FALSE,
                         seed = i * 8 + k,
    )
    write.csv(data, 
              paste('/Users/p-e.poulet/Documents/Aramis/code/notebooks/PPI_CT/simulated_data/Exp3/data_', i, '_', k, '.csv',sep='',collapse=''),
              row.names = FALSE,
    )
  }
}

## Exp 4 : Vary model noise with placebo
for (i in 0:50) {
  for (k in 1:nbRepeats) {
    data <- generateData(nSubj=200,
                         drugEffect = popPars['eStar[1]'],
                         placebo = TRUE,
                         precision = NULL,
                         popParamsRandom = FALSE,
                         seed = i * 8 + k,
    )
    write.csv(data, 
              paste('/Users/p-e.poulet/Documents/Aramis/code/notebooks/PPI_CT/simulated_data/Exp4/data_', i, '_', k, '.csv',sep='',collapse=''),
              row.names = FALSE,
    )
  }
}

## Exp of paper : set different number of subjects, vary predictor's noise (done in notebook)
nbRepeats <- 1000
for (N in c(10, 100, 200, 1000)) {
  for (k in 1:nbRepeats) {
    data <- generateData(nSubj=N,
                         drugEffect = popPars['eStar[1]'],
                         placebo = TRUE,
                         precision = NULL,
                         popParamsRandom = FALSE,
                         seed = N * nbRepeats + k,
    )
    write.csv(data, 
              paste('/Users/p-e.poulet/Documents/Aramis/code/notebooks/PPI_CT/simulated_data/Exp1_paper/data_', N, '_', k, '.csv',sep='',collapse=''),
              row.names = FALSE,
    )
  }
}

## Print subject level expected trajectories
print(
  xyplot(70 * Theta ~ Week | Treatment, groups = SubjId, data
         = dat,
         type = "l",
         col = "lightgrey",
         panel = function(x, y, ...) {
           panel.superpose(x, y, ...)
           panel.average(x, y, col = 'black', type = 'l',
                         lwd = 2, horizontal = FALSE)
           panel.abline(h = seq(20, 30, 2), lty = 2)
         }
  )
)

## Plot subject specific observations
print(
  xyplot(Adas ~ Week | SubjId,
         data = dat,
         subset = SubjId <= 16,
         panel = function(x, y, subscripts, ...) {
           panel.xyplot(x, y, ...)
           panel.xyplot(x, 70 * dat$Theta[subscripts], type =
                          'l')
         },
         scales = list(y = list(relation = "free"))
  )
)