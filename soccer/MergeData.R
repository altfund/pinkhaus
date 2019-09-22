
require(plyr)
#require()

path<-"/Users/eric/B/Soccer/data/hist/raw/"
combo<-data.frame()

for(x in 1994:2014){
folder<-paste0(path,x,"/")
for (i in list.files(folder)){
data<-data.frame()
data<-read.csv(paste0(folder,i))
data$Season<-x
combo<-rbind.fill(combo,data)}}

write.csv(combo,"/Users/eric/B/Soccer/data/hist/new_all-euro-data-1994-2014.csv",row.names=FALSE)