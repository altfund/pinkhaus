wd<-getwd()
setwd("/Users/eric/Documents/Betting/LaLiga")


library(rjags)
library(coda)
library(mcmcplots)
library(stringr)
library(plyr)
library(xtable)
source("plotPost.R")
set.seed(12345)  # for reproducibility

#load("laliga_orig.RData")

laliga<-read.csv("fulldata_20141025.csv")

#convert some columns to correct format to connect to model
laliga$HomeTeam<-as.character(laliga$HomeTeam)
laliga$AwayTeam<-as.character(laliga$AwayTeam)
laliga$Season<-as.character(laliga$Season)
laliga$X.1<-NULL

# -1 = Away win, 0 = Draw, 1 = Home win
laliga$MatchResult <- sign(laliga$HomeGoals - laliga$AwayGoals)

# Creating a data frame d with only the complete match results
d <- na.omit(laliga)
teams <- unique(c(d$HomeTeam, d$AwayTeam))
seasons <- unique(d$Season)

# A list for JAGS with the data from d where the strings are coded as
# integers
data_list <- list(HomeGoals = d$HomeGoals, AwayGoals = d$AwayGoals, HomeTeam = as.numeric(factor(d$HomeTeam,
    levels = teams)), AwayTeam = as.numeric(factor(d$AwayTeam, levels = teams)),
    Season = as.numeric(factor(d$Season, levels = seasons)), n_teams = length(teams),
    n_games = nrow(d), n_seasons = length(seasons))

# Convenience function to generate the type of column names Jags outputs.
col_name <- function(name, ...) {
    paste0(name, "[", paste(..., sep = ","), "]")
}

m3_string <- "model {
for(i in 1:n_games) {
  HomeGoals[i] ~ dpois(lambda_home[Season[i], HomeTeam[i],AwayTeam[i]])
  AwayGoals[i] ~ dpois(lambda_away[Season[i], HomeTeam[i],AwayTeam[i]])
}

for(season_i in 1:n_seasons) {
  for(home_i in 1:n_teams) {
    for(away_i in 1:n_teams) {
      lambda_home[season_i, home_i, away_i] <- exp( home_baseline[season_i] + skill[season_i, home_i] - skill[season_i, away_i])
      lambda_away[season_i, home_i, away_i] <- exp( away_baseline[season_i] + skill[season_i, away_i] - skill[season_i, home_i])
    }
  }
}

skill[1, 1] <- 0 
for(j in 2:n_teams) {
  skill[1, j] ~ dnorm(group_skill, group_tau)
}

group_skill ~ dnorm(0, 0.0625)
group_tau <- 1/pow(group_sigma, 2)
group_sigma ~ dunif(0, 3)

home_baseline[1] ~ dnorm(0, 0.0625)
away_baseline[1] ~ dnorm(0, 0.0625)

for(season_i in 2:n_seasons) {
  skill[season_i, 1] <- 0 
  for(j in 2:n_teams) {
    skill[season_i, j] ~ dnorm(skill[season_i - 1, j], season_tau)
  }
  home_baseline[season_i] ~ dnorm(home_baseline[season_i - 1], season_tau)
  away_baseline[season_i] ~ dnorm(away_baseline[season_i - 1], season_tau)
}

season_tau <- 1/pow(season_sigma, 2) 
season_sigma ~ dunif(0, 3) 
}"

m3 <- jags.model(textConnection(m3_string), data = data_list, n.chains = 3,
n.adapt = 10000)
update(m3, 10000)
s3 <- coda.samples(m3, variable.names = c("home_baseline", "away_baseline","skill", "season_sigma", "group_sigma","group_skill"), n.iter = 40000, thin = 8)
ms3 <- as.matrix(s3)

# The ranking of the teams for the 2012/13 season.
team_skill <- ms3[, str_detect(string = colnames(ms3), "skill\\[5,")]
team_skill <- (team_skill - rowMeans(team_skill)) + ms3[, "home_baseline[5]"]
team_skill <- exp(team_skill)
colnames(team_skill) <- teams
team_skill <- team_skill[, order(colMeans(team_skill), decreasing = T)]
par(mar = c(2, 0.7, 0.7, 0.7), xaxs = "i")
caterplot(team_skill, labels.loc = "above", val.lim = c(0.7, 3.8))

#plotPost(team_skill[, "FC Barcelona"] - team_skill[, "Real Madrid CF"], compVal = 0,
#    xlab = "← Real Madrid     vs     Barcelona →")
    
n <- nrow(ms3)
m3_pred <- sapply(1:nrow(laliga), function(i) {
    #need to account for new teams? repeatedly getting "subscript out of bounds" for ms3[...
  home_team <- which(teams == laliga$HomeTeam[i])
  away_team <- which(teams == laliga$AwayTeam[i])
  season <- which(seasons == laliga$Season[i])
  home_skill <- ms3[, col_name("skill", season, home_team)]
  away_skill <- ms3[, col_name("skill", season, away_team)]
  home_baseline <- ms3[, col_name("home_baseline", season)]
  away_baseline <- ms3[, col_name("away_baseline", season)]

  home_goals <- rpois(n, exp(home_baseline + home_skill - away_skill))
  away_goals <- rpois(n, exp(away_baseline + away_skill - home_skill))
  home_goals_table <- table(home_goals)
  away_goals_table <- table(away_goals)
  match_results <- sign(home_goals - away_goals)
  match_results_table <- table(match_results)

  mode_home_goal <- as.numeric(names(home_goals_table)[ which.max(home_goals_table)])
  mode_away_goal <- as.numeric(names(away_goals_table)[ which.max(away_goals_table)])
  match_result <-  as.numeric(names(match_results_table)[which.max(match_results_table)])
  rand_i <- sample(seq_along(home_goals), 1)

  c(mode_home_goal = mode_home_goal, mode_away_goal = mode_away_goal, match_result = match_result,
    mean_home_goal = mean(home_goals), mean_away_goal = mean(away_goals),
    rand_home_goal = home_goals[rand_i], rand_away_goal = away_goals[rand_i],
    rand_match_result = match_results[rand_i])
})
m3_pred <- t(m3_pred)

laliga_forecast <- laliga[is.na(laliga$HomeGoals), c("Season", "Date", "HomeTeam",
"AwayTeam")] #"Week"
m3_forecast <- m3_pred[is.na(laliga$HomeGoals), ]
laliga_forecast$mean_home_goals <- round(m3_forecast[, "mean_home_goal"], 1)
laliga_forecast$mean_away_goals <- round(m3_forecast[, "mean_away_goal"], 1)
laliga_forecast$mode_home_goals <- m3_forecast[, "mode_home_goal"]
laliga_forecast$mode_away_goals <- m3_forecast[, "mode_away_goal"]
laliga_forecast$predicted_winner <- ifelse(m3_forecast[, "match_result"] ==
    1, laliga_forecast$HomeTeam, ifelse(m3_forecast[, "match_result"] == -1,
    laliga_forecast$AwayTeam, "Draw"))

rownames(laliga_forecast) <- NULL
#print(xtable(laliga_forecast, align = "cccccccccc"), type = "html")

laliga_sim <- laliga[is.na(laliga$HomeGoals), c("Season", "Date", "HomeTeam",
"AwayTeam")] #"Week"
laliga_sim$home_goals <- m3_forecast[, "rand_home_goal"]
laliga_sim$away_goals <- m3_forecast[, "rand_away_goal"]
laliga_sim$winner <- ifelse(m3_forecast[, "rand_match_result"] == 1, laliga_forecast$HomeTeam,
    ifelse(m3_forecast[, "rand_match_result"] == -1, laliga_forecast$AwayTeam,
        "Draw"))

rownames(laliga_sim) <- NULL
#print(xtable(laliga_sim, align = "cccccccc"), type = "html")

matchup <- function(a,b,t){

n <- nrow(ms3)
home_team <- which(teams == a)
away_team <- which(teams == b)
season <- which(seasons == t)
home_skill <- ms3[, col_name("skill", season, home_team)]
away_skill <- ms3[, col_name("skill", season, away_team)]
home_baseline <- ms3[, col_name("home_baseline", season)]
away_baseline <- ms3[, col_name("away_baseline", season)]

home_goals <- rpois(n, exp(home_baseline + home_skill - away_skill))
away_goals <- rpois(n, exp(away_baseline + away_skill - home_skill))

moneyline<-1/c(Home = mean(home_goals > away_goals), Away = mean(home_goals < away_goals), Draw = mean(home_goals == away_goals))

#print(moneyline)
    
goals_payout <- laply(0:6, function(home_goal) {
    laply(0:6, function(away_goal) {
        1/mean(home_goals == home_goal & away_goals == away_goal)
    })
})

colnames(goals_payout) <- paste(a, 0:6, sep = " - ")
rownames(goals_payout) <- paste(b, 0:6, sep = " - ")
goals_payout <- round(goals_payout, 1)
#print(goals_payout)

total_score <- c(Home = mean(home_goals), Away = mean(away_goals), Total = mean(home_goals+away_goals))
#,Greater=mean((home_goals+away_goals)>mean(home_goals+away_goals)))

point_spread <- c(Home = mean(away_goals-home_goals), Away = mean(home_goals-away_goals))

output<-c(Spread=point_spread,Line=moneyline,Goals=total_score)

print(output)
return(output)
}

print(matchup("Sevilla","Valencia","2012"))
print(matchup("Barcelona","Real Madrid","2012"))









setwd(wd)