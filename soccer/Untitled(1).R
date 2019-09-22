wd<-getwd()
basepath <- "/Users/eric/B/Soccer"
setwd(basepath)


library(rjags)
library(coda)
library(mcmcplots)
library(stringr)
library(plyr)
library(xtable)
library(lubridate)
library(stats)
require(PerformanceAnalytics)
source(paste0(basepath,"/model/infrastructure/plotPost.R"))
set.seed(12345)

updatepath<-paste0(basepath,"/data/combo/")
file<-'20141025_euro-fulldata.csv'
data<-read.csv(paste0(updatepath,file[1]),stringsAsFactors=FALSE)

games<-data.frame(
  HomeTeam=data$HomeTeam,
  AwayTeam=data$AwayTeam,
  Div=data$Div,
  Date=data$Date,
  HomeGoals=data$FTHG,
  AwayGoals=data$FTAG,
  Season=data$Season,
  MatchResult=sign(data$FTHG - data$FTAG),
  WinOdds=data$BbAvH,
  DrawOdds=data$BbAvD,
  AwayOdds=data$BbAvA,
  OverOdds=data$BbAv.2.5,
  UnderOdds=data$BbAv.2.5)

games$Date<-as.Date(games$Date,format="%m/%d/%y")

test<-games[-which(games$Season>2013),]
test<-test[-is.na(test$DrawOdds),]




draws<-test[which(test$MatchResult==0),]

