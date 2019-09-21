library(ggplot2)
library(caret)
library(plyr)
library(randomForest)


getData()
{
  temp <- tempfile()
  download.file("http://www.football-data.co.uk/mmz4281/1415/data.zip",temp)
  files <- lapply(unzip(temp),function(x) read.csv(x))
  raw.data <- Reduce(function(x, y) merge(x, y, all=T,sort=F), files, accumulate=F)
  unlink(temp)
  return(raw.data)
}

cleanData(raw.data)
{
  h.odds <- data[c("Date","FTR","B365H","BWH","IWH","LBH","PSH","WHH","SJH","VCH","BbMxH","BbAvH")]
  df <- na.omit(melt(h.odds,id.vars=c("FTR","Date"),variable.name="book",value.name="h.odds"))
  wins <- data.frame(date=as.Date(df$Date,format="%d/%m/%y")
                     ,win=(df$FTR=="H"),book=df$book,pr=1/df$h.odds)
  wins$pr=round(wins$pr,2)
  x <- aggregate(win ~ pr, data=wins, mean)
  x<-round(x,2)
  return(data)
}


model <- function(data)
{
  flds <- createFolds(as.numeric(rownames(data)), k = 3, list = TRUE, returnTrain = FALSE)
  names(flds)[1] <- "train"
  names(flds)[2] <- "cross"
  names(flds)[3] <- "go"
}

ng <- data[c("Div","FTR","BbMxH","BbAvH","BbMxD","BbAvD","BbMxA","BbAvA","BbMx.2.5","BbAv.2.5")]

# 
# ggplot(df, aes(x=h.odds, colour=book)) + 
#   geom_density() +
#   #geom_histogram(binwidth=.5, colour="black", fill="white") + 
#   facet_grid(FTR ~ .) +
#   geom_vline(data=cdat, aes(xintercept=h.odds.mean, colour=book),
#              linetype="dashed", size=1)
