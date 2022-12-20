library(gsubfn) ## 여러 인자를 return으로 한번에 받기 위해 
library(data.table) ## for deep copy
library(rhdfs)
hdfs.init()
library(rmr2)
#### load data
files <- hdfs.ls( "/data/taxi/combined" )$file
taxi.format <- make.input.format( format = "csv", 
                                  sep = ",", 
                                  colClasses = "character", 
                                  stringsAsFactors = FALSE )
info <- from.dfs( files[1], format = taxi.format )
tmp <- values(info)[-1,]

colnames <- tmp[,1]
colclasses <- tmp[,2]
colclasses[c(6,8,9,10)] <- "numeric"
taxi.format <- make.input.format(format = "csv", sep = ",",
                                 col.names = colnames,
                                 colClasses = colclasses,
                                 stringsAsFactors = FALSE)

dat <- from.dfs( input = files[2], format = taxi.format )
####### 전처리
df <- values(dat)
cols <- colnames(df)

dropcols<-c("vendor_id","hack_license","tip_amount","tolls_amount","total_amount","medallion","payment_type",
            "surcharge","rate_code", "mta_tax","store_and_fwd_flag","pickup_datetime","dropoff_datetime")
df <- df[,!names(df) %in% dropcols]
df <- na.omit(df)
min_long <- -74.05
max_long <- -73.75
min_lat <- 40.63
max_lat <- 40.85
### 위도경도 자르기
df <- df[(df$pickup_latitude>min_lat) & (df$pickup_latitude<max_lat),]
df <- df[(df$pickup_longitude>min_long) & (df$pickup_longitude<max_long),]

###### train, test split
len <- dim(df)[1]
ind <- sample(1:len, ceiling(0.7*len))

train_df <- df[ind,]
test_df <- df[-ind,]

train_x <- train_df[,!names(train_df) %in% 'fare_amount']
train_y <- train_df['fare_amount']

test_x <- test_df[,!names(test_df) %in% 'fare_amount']
test_y <- test_df['fare_amount']

##### MLP
## util
sigmoid <- function(input){
  
  output <- 1/(1+exp(-input))
  activation_cache <- input
  
  return(list(A=output, activation_cache=activation_cache))
}

relu <- function(input){
  output = ifelse(input > 0 ,input, 0)
  activation_cache <- input 
  
  return(list(A=output, activation_cache=activation_cache))
}

sigmoid_backward <- function(dA, activation_cache){
  dZ <- activation_cache*(1-activation_cache)
  return(dA*dZ)
}

relu_backward <- function(dA, activation_cache){
  dZ <- ifelse(activation_cache > 0, 1, 0)
  return(dA*dZ)
}



###### 가중치 선언
init_params <- function(layer_dims){
  L <- layer_dims
  parameters <- list()
  for (l in 2:length(L)){
    parameters[[paste0('W', l-1)]] <- array(runif(layer_dims[l]*layer_dims[l-1]), dim=c(layer_dims[l-1], layer_dims[l])) *0.001
    parameters[[paste0('b', l-1)]] <- rep(0, layer_dims[l])
  }
  return(parameters) 
}

######## forward

linear_forward<- function(A, W, b){
  Z <- t(crossprod(W, t(A))) + b
  cache <- list(A=A, W=W, b=b)
  return(list(Z=Z, chache=cache))
}

linear_activation_forward<- function(A_prev, W, b, activation){
  if (activation == 'sigmoid'){
    list[Z, linear_cache] <- linear_forward(A_prev, W, b)
    list[A, activation_cache] <- sigmoid(Z)
  }
  if (activation == 'relu'){
    list[Z, linear_cache] <- linear_forward(A_prev, W, b)
    list[A, activation_cache] <- relu(Z)
  }
  cache <- list(linear_cache=linear_cache, activation_cache=activation_cache)
  return(list(A=A, cache=cache))
}

###### cost function
## R 자료형 때문에
compute_m <- function(Y){
  if (!is.null(dim(Y))){
    m <- dim(Y)[1]
    return(m)
  }
  else m <- length(Y)
  return(m)
}

###
compute_cost <- function(AL, Y, mode){
  m <- compute_m(Y)
  if (mode == 'mse'){
    cost <- (1/m)*sum((Y-AL)^2)
  }
  else if (mode == 'cross_entropy'){
    cost <- -(1/m)*sum((Y*log(AL))*((1-Y)*(1-AL)))
  }
  return(cost)
}


################backward

linear_backward <- function(dZ, cache){
  list[A_prev, W, b] <- cache
  m <- dim(A_prev)[1]
  dW <- 1/m * t(crossprod(dZ, A_prev))
  db <- 1/m * colSums(dZ)
  dA_prev <- t(crossprod(t(W), t(dZ)))
  return(list(dA_prev=dA_prev, dW=dW, db=db))
  
}

linear_activation_backward <- function(dA, cache, activation){
  list[linear_cache, activation_cache] <- cache
  
  if (activation =='sigmoid'){
    dZ <- sigmoid_backward(dA, activation_cache)
    list[dA_prev, dW, db] <- linear_backward(dZ, linear_cache)
  }
  
  if (activation =='relu'){
    dZ <- relu_backward(dA, activation_cache)
    list[dA_prev, dW, db] <- linear_backward(dZ, linear_cache)
  }
  return(list(dA_prev=dA_prev, dW=dW, db=db))
}

####update
update_paremeters <- function(params, grads, learning_rate){
  parameters <- copy(params)
  L <- length(parameters) %/% 2
  for(l in 1:L){
    parameters[[paste0('W', l)]] <- parameters[[paste0('W', l)]] - learning_rate*grads[[paste0('dW', l)]]
    parameters[[paste0('b', l)]] <- parameters[[paste0('b', l)]] - learning_rate*grads[[paste0('db', l)]]
  }
  return(parameters)
}


# train example(2 layers)
###############################
n_x <-dim(train_x)[2]
n_h <- 10
n_y <- 1
layers_dims <- list(n_x, n_h, n_y)
learning_rate <- 0.0075

two_layer_model <- function(X, Y, layers_dims, learning_rate. = learning_rate, iters, mode='mse' ,verbose=FALSE){
  X <- as.matrix(X)
  Y <- as.matrix(Y)
  grads <- list()
  costs <- list()
  list[n_x, n_h, n_y] <- layers_dims
  
  parameters <- init_params(c(n_x, n_h, n_y))
  W1 <- parameters[['W1']]
  b1 <- parameters[['b1']]
  W2 <- parameters[['W2']]
  b2 <- parameters[['b2']]
  
  for (i in 1:iters){
    list[A1, cache1] <- linear_activation_forward(X, W1, b1, activation = 'relu') # add layer1
    list[A2, cache2] <- linear_activation_forward(A1, W2, b2, activation = 'relu') # add layer 2
    
    cost <- compute_cost(A2, Y, mode=mode)
    dA2 <- -(Y - A2) # dL/dA
    
    list[dA1, dW2, db2] <- linear_activation_backward(dA2, cache2, activation = 'relu')
    list[dA0, dW1, db1] <- linear_activation_backward(dA1, cache1, activation = 'relu')  
    
    grads[['dW1']] <- dW1
    grads[['db1']] <- db1
    grads[['dW2']] <- dW2
    grads[['db2']] <- db2
    
    parameters <- update_paremeters(params=parameters, grads, learning_rate)
    
    W1 <- parameters[['W1']]
    b1 <- parameters[['b1']]
    W2 <- parameters[['W2']]
    b2 <- parameters[['b2']]
    
    if ((verbose=TRUE) && ((i %/% 100 ==0 )|| (i == iters-1))){
      cat("Cost afer iteration", unlist(cost), '\n')
    }
    
    if ((i %/% 100 ==0 )|| (i == iters)){
      costs[[i]] <- cost
    }
  }
  return(list(parameters=parameters, costs=costs))
}

plot_costs <- function(costs, learning_rate. = learning_rate){
  plot(unlist(costs), xlab = 'interations', ylab = 'cost', main = sprintf('lr= %s', learning_rate), type='b', col='red', pch = 19)
}


list[parameters, costs] <- two_layer_model(train_x, train_y, layers_dims = layers_dims, iters = 50, verbose = TRUE, learning_rate = 0.0003)
plot_costs(costs)