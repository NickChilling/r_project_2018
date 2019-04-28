# 1. 建立garch模型计算波动率
# 2. 处理数据集
# 3. 建立神经网络模型进行预测
# 4. 与bs模型效果做对比


# ### 数据集处理
# 期权价格、资产价格、行权价格，是通过同花顺软件获得的CSV文件，
# 分别在asset 和option 两个表中



# 1.1 通过构造request请求，以及通过css对网页节点选择，获取无风险利率(shibor3月利率)
library("rvest")
#抓取单页数据
craw_one<-function(page){
  url <- 'http://data.eastmoney.com/shibor/
  shibor.aspx?m=sh&t=99&d=99225&cu=cny&type=009020&p='
  t_url <-paste(url,page)
  webpage <- read_html(t_url)
  data <- html_nodes(webpage,'td')
  data <- html_text(data)
  k <- t(matrix(data,nrow =3))
  frame1 <- data.frame(date = k[,1],rate = k[,2],
  	stringsAsFactors = FALSE)
  return (frame1)
}

#多页数据抓取
crawl_interest<- function(){
  irate <- craw_one(1)
  for(i in 2:12)
  {
    irate <- rbind(irate,craw_one(i))
  }
  irate$date<- as.Date(irates$date)
  irate$rate <-as.numeric(as.character(irates$rate))
  return(irate)
}
irates <-crawl_interest()
class(irates)
write.csv(irates,file="irates.csv")



### 数据清洗部分
library(stringr)
#read_data
option1 <- read.csv('option.csv',sep = "\t",
	stringsAsFactors = FALSE)
irates <- read.csv('irates.csv',stringsAsFactors = FALSE)

#数值类型转换
option1$time<- as.Date(as.character(option1$time))
option1$close<- as.numeric(option1$close)

asset$time <- as.Date(asset$time)
names(asset)[4] <-'asset.close'
asset <- asset[!is.na(asset$asset.close),]#删除NA

#增加敲定价格
option1$strike<-str_sub(option1$asset_name,-4,-1)
option <-option1[str_sub(option1$asset_name,6,6)== 
rep("购",dim(option1)[1]),] # 删除看跌期权的数据
option <- option[!is.na(option$close),]
head(option)

option$strike<- as.numeric(option$strike)
option$strike<-option$strike/1000

head(asset)

#定义将两表拼接的函数
concat <-function(df1,df2,c1,c2,c3){
  row1 <- dim(df1)[1]
  row2 <- dim(df2)[1]
  colum <- numeric(row1)
  flag <- FALSE
  
  for(i in 1:row1){
    for(j in 1:row2){
      if((df1[[c1]][i]==df2[[c2]][j]))
      {
        colum[i] <- df2[[c3]][j]
        flag<-TRUE
      }
    }
    if(flag==TRUE)
      flag <-FALSE
    else{
      print(i)
      return (FALSE)
    }
  }
  return(colum)
}

asset.close <-concat(option,asset,3,3,4)
asset.sigma <-concat(option,asset,3,3,5)
irate <-concat(option,irates,3,2,3)

#判定长度是否相等
dim(option)[1]==length(asset.close)
length(asset.close)==length(irate)


final_data <-cbind(option,asset.close,irate,asset.sigma)
head(final_data)
write.csv(final_data,"dataset.csv")




####2 建立garch模型，获取波动率
library(tseries)
library(forecast)
library(fGarch)

asset <- read.csv('asset.csv',sep = "\t",stringsAsFactors = FALSE)
asset <- asset[!is.na(asset$close),]
log.return<-diff(log(asset$close))


# 检验序列是否平稳，否则需要差分

ndiffs(log.return)

plot(log.return,type = 'l')

# 正态检验
jarque.bera.test(log.return)

# 均值检验,不显著 无法假定真实均值！=0
t.test(log.return)
hist(log.return)

# 自相关检验，存在自相关 应建立简单的AR模型
par(mfrow=c(2,1))
acf(log.return)
pacf(log.return)

# 建立ar模型，确认阶数
mm = ar(log.return,method = 'mle')
a <- mm$resid
#plot(c(0:12),mm$aic+mm,type = 'h')

# 检验残差 p<=0.05 说明有很强的ARCH效应
par(mfrow=c(2,1))
acf(a[8:221]^2)
pacf(a[8:221]^2)
for(i in 1:10)
{
  print(Box.test(a^2,lag=i,type = 'Ljung'))
}


#建立arma(6,0),GARCH(1,1)
g.1.1 = garchFit(~arma(6,0)+garch(1,1),
	data = log.return,trace = F)
summary(g.1.1)

gresi = residuals(g.1.1,standardize=T)
sigma <- g.1.1@sigma.t

# 再一次检验残差，显示没有arch效应
par(mfrow=c(2,1))
acf(gresi,lag =24)
pacf(gresi,lag = 24)
for(i in 1:10){
  print(Box.test(gresi^2,lag=i)) 
}


# 刻画置信区间，可视化建模效果
upp <- -0.00089+2*sigma
low <- -0.00089-2*sigma
par(mfrow=c(1,1))
plot(log.return,type = 'l')
lines(upp,lty=2,col = 'red')
lines(low,lty=2,col = 'red')
asset <-asset[-1,]
asset<- cbind(asset,sigma)
head(asset)




#####搭建神经网络模型
library(keras)
library(tibble)

data.ori <- read.csv("dataset.csv",stringsAsFactors = FALSE)
data.ori$irate<-data.ori$irate/100

head(data.ori)


# 区分数据集

set.seed(1)
ind <- sample(2,nrow(data.ori),replace=TRUE,prob = c(0.7,0.3))
train.ori<-data.ori[ind==1,]
test.ori<-data.ori[ind==2,]
head(train.ori)
train_data <- train.ori[,6:9]
train_label <-train.ori[[5]]
test_data <-test.ori[,6:9]
test_label <-test.ori[[5]]
head(train.ori)

train_data <-scale(train_data)
col_means_train <-attr(train_data,"scaled:center")
col_stds_train<-attr(train_data,"scaled:scale")
test_data <-scale(test_data,center=col_means_train,scale=col_stds_train)  # 使用训练集的均值和标准差归一化测试集
#搭建模型
head(test_data)

loss_function <-function(y_true,y_predict){
  return (mean(abs(y_true-y_predict)/y_true))
}
b_model<-function(){
  
  model<-keras_model_sequential() %>%
  layer_dense(units = 30,activation = "sigmoid",
              input_shape = dim(train_data)[2]) %>%
  layer_dense(units = 30,activation="relu") %>%
  layer_dense(units = 30, activation="sigmoid") %>%
  layer_dense(units = 30, activation="relu") %>%
  layer_dense(units = 30, activation="relu") %>%
  layer_dense(units = 30, activation="relu") %>%
  layer_dense(units = 1)
  
  model %>% compile(
    loss = loss_mean_absolute_percentage_error,
    optimizer = optimizer_adam(lr=0.001),
    metrics = list(metric_mean_absolute_percentage_error)
  )
  
  model
}

model<-b_model()
model%>% summary()

## 定义打印损失函数频次
print_dot_callback<-callback_lambda(
  on_epoch_end = function(epoch,logs){
    if(epoch%%80 == 0) cat("\n")
    cat(".")
  }
)

# 定义训练次数
epochs <-600

# 开始训练
history <-model %>% fit(
  train_data,
  train_label,
  epochs = epochs,
  validation_split = 0.2,
  verbose=0,
  batch_size = 200,
  callbacks = list(print_dot_callback)
  
)

library(ggplot2)
summary(history)
plot(history)

# 使用测试集验证
test_predictions <-model %>% predict(test_data)
loss_function(test_label,test_predictions)
plot(test_label,type='l',xlab = 'time series',ylab = 'price')  
# 0.2without early stop

lines(test_predictions,lty=2,col = 'blue')
plot(abs(test_label-test_predictions))


# 检验bs模型预测效果
library(RND)
bsm = numeric(dim(test.ori)[1])
for(i in 1:dim(test.ori)[1])
    {
      bsm[i] <-price.bsm.option(test.ori[i,7],test.ori[i,6],test.ori[i,8],1,test.ori[i,9],0)$call
}

plot(test_label,type='l',main= "fit curve",xlab = "time series",ylab="price")  
lines(bsm,lty=2,col = 'red')

k<-loss_function(test_label,bsm)# 0.6