#Load
hist <- read.csv("~/Downloads/Master - Sheet1.csv", stringsAsFactors=FALSE)
key <- read.csv("~/Downloads/SoccerTeams_key - Key.csv", stringsAsFactors=FALSE)
res.name <- "/all-euro-data-1994-2014.csv"
results <- read.csv(paste0("/Users/eric/B/Soccer/data/hist",res.name)
                    , stringsAsFactors=FALSE)

#clean


#Functions
match_team <- function(team,key) {
  #takes a team name and returns the bovada and db versions of the name
}

#Spread: [Pick (-225), Pick (+170)]
#Moneyline: [-110, +260, +250]
#Total: [2½, (EVEN)o, (-130)u]

#Format
df <- data.frame(date <- as.Date(hist$pull_date,format="%m/%d/%Y"),
                 home.team <- "Arsenal",
                 away.team <- "Liverpool",
                 league <- "England"
                 moneyline.home <- 1,
                 moneyline.away <- 1,
                 moneyline.draw <- 1,
                 total.total <- 1,
                 total.over <- 1,
                 total.under <- 1,
                 spread.spread <- 1,
                 spread.home <- 1,
                 spread.away <- 1
  )

m<-unlist(strsplit(gsub("]","",gsub("\\[","",hist$moneyline)),", "))
#moneyline.h
#moneyline.a
#moneyline.d

#convert lines to readable format? or function for that
#total.total
#total.over
#total.under
#league
#

#take 75% of the data for training and 25% for cross-validation

#Analyze
#


#Present