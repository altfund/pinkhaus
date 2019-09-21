require(gdata)
require(plyr)

basepath<-"/Users/eric/B/Soccer"
setwd(basepath)




readZipURL<-function(urlPath,basefn){
xlsFile <- paste0(basefn,".xls")
zipFile <- paste0(basefn,".zip")
download.file(paste0(urlPath,zipFile),zipFile)
unzip(zipFile)
gdata::read.xls(xlsFile)
unlink(zipFile)

df=data.frame()
for (x in 1:sheetCount(xlsFile)){
df=rbind.fill(df,read.xls(xlsFile,sheet=x,header=TRUE))
}
df
}

basefn <- "all-euro-data-2014-2015"
urlPath <- "http://www.football-data.co.uk/mmz4281/1415/"

#newdata <- readZipURL(urlPath,basefn)

url <- "http://www.football-data.co.uk/mmz4281/1415/data.zip"

zipdir <- tempfile()
dir.create(zipdir)
#download.file(url,zipdir, mode="wb")
download.file(url, temp <- tempfile(fileext = ".zip"))
unzip(temp,exdir=zipdir)
files<-list.files(zipdir)
newdata<-data.frame()
for (x in files){
	newdata<-rbind.fill(newdata,read.csv(paste(zipdir,x,sep="/")))
}
newdata$Season<-2015

extantpath<-paste0(basepath,"/data/recent/")
extantdata<-read.csv(paste0(extantpath,basefn,".csv"))


#for some reason the identical test doesn't seem to be working
#doesn't matter too much since it's just more data pulls, but worth checking out.
if(!identical(extantdata,newdata)){
	write.table(newdata,file(paste0(extantpath,basefn,".csv"),"w"),sep=",",row.names=FALSE)
	histdata<-read.csv(paste0(basepath,"/data/hist/all-euro-data-1994-2014.csv"))
	updated <- rbind.fill(newdata,histdata)
	recentname<-paste0(format(Sys.Date(),format="%Y%m%d"),"_euro-fulldata")
	updatepath<-paste0(basepath,"/data/combo/to_process/",recentname,".csv")
	write.table(updated,updatepath,sep=",",row.names=FALSE)
	}
	
quit(save="no")
	
