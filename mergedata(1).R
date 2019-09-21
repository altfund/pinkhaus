wd<-getwd()

compile<-function(path){
	wd<-getwd()
	setwd(paste0(path,"/fulldata/",collapse=NULL))
	
	files<-c(list.files())
	data<-data.frame()
	
	for(i in 1:length(files)){
		str<-c(unlist(strsplit(files[i],"[.]")))
		season<-unlist(strsplit(str[1],"_"))[2]
		file<-read.csv(files[i])
		print(str(file))
		#file<-na.omit(file)
		file$Season<-season
		header<-colnames(file)
		if(length(data)>0){
		for(j in 1:length(header)){
			if((header[j] %in% colnames(data))==FALSE){
				data$new<-NA
				names(data)[names(data)=="new"] <- header[j]
				}
		}
		for(k in 1:length(data)){
			if((colnames(data)[k] %in% header)==FALSE){
				file$new<-NA
				names(file)[names(file)=="new"] <- colnames(data)[k]
				}
		}
		}
		#print(str(data))
		#print(str(file))
		#print(season)
		#print(names(data))
		#print(names(file))
		data<-rbind(data,file)
		#print(names(data))
		#print(season)
		
	}
	setwd(path)
	#data<-data[which(any(apply(data,1,is.na)))]
	full<-write.csv(data, file="fulldata.csv")
	full
}

simple<-function(path){
	data<-data.frame()
	wd<-getwd()
	setwd(paste0(path,"/fulldata/",collapse=NULL))
	
	files<-c(list.files())
	data<-data.frame()
	
	for(i in 1:length(files)){
		str<-c(unlist(strsplit(files[i],"[.]")))
		season<-unlist(strsplit(str[1],"_"))[2]
		file<-read.csv(files[i])
		#print(str(file))
		#file<-na.omit(file)
		file$Season<-season
		new<-data.frame(file$HomeTeam,file$AwayTeam,file$FTHG,file$FTAG,file$Season)
		data<-rbind(data,new)
		}
		
	setwd(path)
	#data<-data[which(any(apply(data,1,is.na)))]
	use<-write.csv(data, file="usedata.csv")		
	return(use)
}

use<-simple("/Users/eric/Documents/Betting/Soccer/England/EPL")
#full<-compile("/Users/eric/Documents/Betting/Soccer/England/EPL")

#setwd(wd)