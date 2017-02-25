# This data set contains information on loans from Lending Club, and has ~70 different variables. 
# I know that this data set came from Kaggle, but I had never looked at it closely.
# I am not going to be the guy that copies the winning python script that uses tensorflow or something
# to build a model that can predict charge-off with an ROC of 0.999. I want to focus on exploratory 
# data analysis, hypothesis generation, and find something interesting that will be useful for
# business intelligence. What would a lender most care about? Lenders have two principal concerns: How
# can I make the most money for my invested capital, and how can I prevent charge-off. 

# I need two response variables, one for the return on invested capital, and another for the 
# loan status (paid or charged-off). The loan status is already in the data set ("loan_status"). 
# How can we measure loan profitability? With net annualized return or return on equity. 
# I need a few variables to calculate this:
# length of the loan period - "issue_d" is the month loan was funded and "last_pymnt_d" for the month
# that the last payment was received. Now I need a formula for annual yield. I will just use:
# APY = ( (principal + gain) / principal ) ^ (12/months) - 1
# now I need to calculate the principal and profit for each loan.
# Let's look at the relevant variables for a few loans...

# look at the data
library(dplyr)
library(stringr)
library(ggplot2)
library(caret)
lc = readr::read_csv("loan.csv")
dim(lc)
# look at na values
na_by_category = lapply(lc, FUN = function(x) sum(is.na(x) / length(x))) 
# do any categories have over 50% na?
na_by_category[na_by_category > 0.5]
# what is the typical loan status
table(lc$loan_status)
# view the variables relevant for calculating APY for fully paid and charged off loans
lc %>% filter(loan_status == "Fully Paid") %>% 
  select(c(id, funded_amnt, funded_amnt_inv,
           out_prncp, out_prncp_inv, recoveries, total_pymnt,
           total_pymnt_inv, total_rec_int, total_rec_late_fee, total_rec_prncp,
           recoveries)) %>% slice(1:5)
lc %>% filter(loan_status == "Charged Off") %>% 
  select(c(id, funded_amnt, funded_amnt_inv,
           out_prncp, out_prncp_inv, recoveries, total_pymnt,
           total_pymnt_inv, total_rec_int, total_rec_late_fee, total_rec_prncp,
           recoveries)) %>% slice(1:5)
# The profit is simple to calculate and APY is (total_pymnt/funded_amnt) ^ (12/months) - 1
# let's see how to find out how many months the loan was active
lc %>% filter(loan_status == "Fully Paid") %>% 
  select(c(id, funded_amnt, issue_d, last_pymnt_d)) %>% slice(1:5)
########### remember to put these in kable for markdown file
lc %>% filter(loan_status == "Charged Off") %>% 
  select(c(id, funded_amnt, issue_d, last_pymnt_d)) %>% slice(1:5)
# let's use lubridate to find out how long each loan was active
library(lubridate)
# coerce to the date format
lc[] = lc %>% purrr::map_if(names(.) %in% c("issue_d", "last_pymnt_d"), ~ dmy(stringr::str_c("01-", .)))
# find the number of months each loan was active
# this will only work for completed loans so remove currents   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# may have to start earlier
lc = filter(lc, last_pymnt_d < "2015-08-02")
lc = mutate(lc, months_active = month(last_pymnt_d) - month(issue_d) + 
              (12 * (year(last_pymnt_d) - year(issue_d))))
table(lc$loan_status)
lc = filter(lc, loan_status != "Late (31-120 days)")

# a few loans were open for zero months, lets set those to one mmonth
lc$months_active[lc$months_active == 0] = 1
# now calculate the APY
lc = mutate(lc, apy = ( (total_pymnt/funded_amnt) ^ (12/months_active) ) - 1  )
# remove any entries that do not have values for entries used to calculate the apy
lc = lc[lc %>% select(issue_d, last_pymnt_d, total_pymnt, funded_amnt, months_active, apy) %>% 
          complete.cases(), ]
max(lc$apy)
median(lc$apy)
mean(lc$apy)
lc %>% ggplot(aes(apy)) + geom_density(color = "blue", fill = "light blue") 
table(lc$loan_status)
# there are a few outliers, I will remove them - about the top 0.01%
table(lc$apy > 0.5)
lc = filter(lc, apy < 0.5)
lc %>% ggplot(aes(apy)) + geom_density(color = "blue", fill = "light blue") 

# i will also make a normalized apy variable
# function to standardize a variable
standardizer = function(x) {
  preprocessParams = preProcess(data.frame(x), method=c("center", "scale", "YeoJohnson"))
  transformed_data = predict(preprocessParams, data.frame(x))
  transformed_data[[1]]
}

lc$transformed_apy = standardizer(lc$apy)
lc %>% ggplot(aes(transformed_apy)) + geom_density(color = "blue", fill = "light blue") 


# lets find some interesting variables to see how they relate to apy
# first let's remove those with >90% NAs
names(na_by_category[na_by_category > 0.9])
variable_list = names(lc)[!(names(lc) %in% names(na_by_category[na_by_category > 0.9]))]
# print the variables
cat(variable_list, sep = "\n# ")

# make a list of variables i want to keep in the model
model_variables = c()

# id 
# member_id
# loan_amnt let's start here
# function to plot numeric variables
plot_variable = function(name, bin_n = 50) {
  z = which(names(lc) == name)
  str(lc[[z]])
  table(is.na(lc[[z]]))
  ggplot(aes(x = get(names(lc)[z]), y = apy), data = lc) + geom_hex(bins=bin_n) + stat_smooth(color="red") +
    scale_fill_gradientn(colours=c("#dafffe", "dark blue"),name = "Frequency",na.value=NA) + xlab(name)
}
plot_variable("loan_amnt")

model_variables = append(model_variables, "loan_amnt")

# funded_amnt - redundant
cor(lc$loan_amnt, lc$funded_amnt)

# term 
ggplot(aes(x = term, y = apy), data = lc) + 
  geom_boxplot(outlier.alpha = 0.1, outlier.size = 0.2) + stat_summary(color= "red")
model_variables = append(model_variables, "term")

# int_rate - great variable
plot_variable("int_rate")
model_variables = append(model_variables, "int_rate")



# installment
plot_variable("installment", 100)
model_variables = append(model_variables, "installment")
# let's make a new variable that is installment divided by income
f


# grade
ggplot(aes(x = grade, y = apy), data = lc) + geom_boxplot(outlier.alpha = 0.1, outlier.size = 0.2) 
model_variables = append(model_variables, "grade")
# can we build a model from just the grade?
library(modelr)
mod = lm(apy ~ grade, data = lc)
grid = lc %>% data_grid(grade) %>% add_predictions(mod, "apy")
ggplot(lc, aes(grade, apy)) + geom_boxplot(outlier.alpha = 0.1, outlier.size = 0.2) + 
  geom_point(data = grid, color = "red", size = 3) + ggtitle("Linear Model Predictions")
# we probably only want to fund on a grade loans


# sub_grade
ggplot(aes(x = sub_grade, y = apy), data = lc) + geom_boxplot(outlier.alpha = 0.1, outlier.size = 0.2) 
# i will just keep grade for now


# emp_title 
# I am going to make this into a new feature
lc$emp_title = trimws(tolower(lc$emp_title))
# let's figure out how common a title is, chief fiancial officer is a rare title, and 
# possibly someone who can pay back a loan
lc$emp_title[1:20] # it looks like most people put down the name of their employer, not the job title
job_lookup = as.data.frame(table(lc$emp_title))
job_lookup = rename(job_lookup, job_title_frequency = Freq)
lc = left_join(lc, job_lookup, by = c("emp_title" = "Var1"))
plot_variable("job_title_frequency")
# my new variable can explain some of the variance, I will add it
model_variables = append(model_variables, "job_title_frequency")
# what else can i do with this job title variable?

# count misspelled words in job title - someone who writes "Stock merket traider" for his job is probably
# not worth the risk

job_lookup$Var1 = stringr::str_replace_all(job_lookup$Var1,"[[:punct:]]", "")
job_lookup$Var1 = stringr::str_replace_all(job_lookup$Var1,"[[:digit:]]", " ")
job_lookup$Var1 = trimws(job_lookup$Var1)
library(parallel)
cores <- detectCores() - 1
library(hunspell)
job_lookup$Var1 = as.character(job_lookup$Var1)
spell_check = function(x) {
    x = unlist(strsplit(x, split = " "))
    x = x[x != ""]
    sum(!hunspell_check(x))
}
job_lookup$n_misspelled = unlist(mclapply(job_lookup$Var1, spell_check, mc.cores = cores))

lc = left_join(lc, job_lookup %>% select(Var1, n_misspelled), by = c("emp_title" = "Var1"))


plot_variable("n_misspelled")
# make a model
mod = lm(apy ~ n_misspelled, data = lc)
grid = lc %>% data_grid(n_misspelled) %>% add_predictions(mod, "apy")
ggplot(lc, aes(jitter(n_misspelled), apy)) + geom_hex() + 
  stat_smooth(color = "red") 
# this doesn't make any sense so I not include it

# emp_length
ggplot(lc, aes(emp_length, apy)) + geom_boxplot() + 
  stat_summary(color = "red") 
# we'll let 
# I'm going to convert this from a factor to an integer 
lc$emp_length = unlist((as.integer(str_match_all(lc$emp_length, "[0-9]+"))))
plot_variable("emp_length")
variable_list = append(variable_list, "emp_length")


# home_ownership
ggplot(lc, aes(home_ownership, apy)) + geom_boxplot() + 
  stat_summary(color = "red") 
# this is interesting, it looks like the homeless pay back their loans the most! let's keep the variable
variable_list = append(variable_list, "home_ownership")

# annual_inc
plot_variable("annual_inc")
# take a look at the guy with a 9 mil annual income who didn't pay back his $8,000 loan
lc[which(lc$annual_inc == max(lc$annual_inc, na.rm = TRUE)), ] %>% select(id, loan_amnt, annual_inc)
variable_list = append(variable_list, "annual_inc")
# i will also create a transformed annual income to make a model
lc$transformed_inc = standardizer(lc$annual_inc)
summary(lm(transformed_apy ~ transformed_inc, data = lc))
ggplot(lc, aes(transformed_inc, transformed_apy)) + geom_hex() + 
  stat_smooth(color = "red") 
# a really clear relationship


# verification_status 
ggplot(lc, aes(verification_status, apy)) + geom_boxplot() + 
  stat_summary(color = "red") 
variable_list = append(variable_list, "annual_inc")
# again this is weird because it looks like people who would be riskier (homeless, unverified income)
# actually perform better


# pymnt_plan
table(lc$pymnt_plan) # i'll ignore this

# desc - here we go again, so much fun could be had with this.. what do these look like...
lc$desc[2]
# i need to clean these up and remove the html, etc... 
lc$desc_transformed = as.character(tolower(lc$desc))
things_to_remove = c("[[:punct:]]", "[[:digit:]]", "borrower added on", "<br>", ">")
for(i in seq_along(things_to_remove)){
  lc$desc_transformed = str_replace_all(lc$desc_transformed, things_to_remove[i]," ")
}
lc$desc_transformed = trimws(lc$desc_transformed)
# now let's check it
lc$desc_transformed[2]
# let's do a variable for the length of the description
lc$desc_nchar = nchar(lc$desc_transformed)
lc$desc_nchar[is.na(lc$desc_nchar)] = 0
plot_variable("desc_nchar")
# look like a good variable
variable_list = append(variable_list, "desc_nchar")

# look at this...
print(lc$desc_transformed[8])
# what if i counted the number of times "$" appears, would you give money to a guy that says:
# "need $$$$$$$ now!"
lc$dollar_signs = str_count(lc$desc_transformed, pattern="\\$")
plot_variable("dollar_signs")
# i have to see the max, with 50 $'s
lc$desc_transformed[which(lc$dollar_signs == max(lc$dollar_signs, na.rm = TRUE))]
# oh brother
variable_list = append(variable_list, "dollar_signs")

# let's try my spell checker again

lc$desc_misspelled = unlist(mclapply(lc$desc_transformed, spell_check, mc.cores = cores))
plot_variable("desc_misspelled")
# i'm not sure this variable makes sense


# purpose - this looks interesting, wedding loans are paid back
ggplot(lc, aes(purpose, apy)) + geom_boxplot() + 
  stat_summary(color = "red") + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
variable_list = append(variable_list, "purpose")



# title
lc$title[1:10]
# let's make a variable for length and check misspellings again
lc$title_nchar = nchar(lc$title)
lc$title_nchar[is.na(lc$title_nchar)] = 0
plot_variable("title_nchar") # great variable
variable_list = append(variable_list, "title_nchar")

lc$title_transformed = str_replace_all(lc$title, "[^[:alnum:]]", " ")
lc$title_transformed = str_replace_all(lc$title, "[[:digit:]]", " ")
lc$title_misspelled = unlist(mclapply(lc$title_transformed, spell_check, mc.cores = cores))
ggplot(lc, aes(title_misspelled, apy)) + geom_hex() + 
  stat_summary(color = "red") 
# again, this doesn't really make sense so I will not keep this variable


# zip_code - i will ignore this for now

# dti
plot_variable("dti") # great variable
variable_list = append(variable_list, "dti")

# delinq_2yrs
plot_variable("delinq_2yrs")
variable_list = append(variable_list, "delinq_2yrs")

# earliest_cr_line
# let's get the year and look at a plot
lc$earliest_cr_line_transformed = as.numeric(str_split(lc$earliest_cr_line, "-", simplify = TRUE)[, 2])
plot_variable("earliest_cr_line_transformed")
# great variable, let's keep it
variable_list = append(variable_list, "earliest_cr_line_transformed")


# inq_last_6mths
plot_variable("inq_last_6mths")
variable_list = append(variable_list, "inq_last_6mths")


# mths_since_last_delinq # the na's will be important - let's bin this and encode as a factor
lc$mths_since_last_delinq_transformed = "NONE"
lc$mths_since_last_delinq_transformed[lc$mths_since_last_delinq > 24] = "OVER 24"
lc$mths_since_last_delinq_transformed[lc$mths_since_last_delinq <= 24] = "UNDER 24"
ggplot(lc, aes(mths_since_last_delinq_transformed, apy)) + geom_boxplot() + 
  stat_summary(color = "red") 
variable_list = append(variable_list, "mths_since_last_delinq_transformed")


# mths_since_last_record - I'll do the same to this
lc$mths_since_last_record_transformed = "NONE"
lc$mths_since_last_record_transformed[lc$mths_since_last_record > 24] = "OVER 24"
lc$mths_since_last_record_transformed[lc$mths_since_last_record <= 24] = "UNDER 24"
ggplot(lc, aes(mths_since_last_record_transformed, apy)) + geom_boxplot() + 
  stat_summary(color = "red") 
variable_list = append(variable_list, "mths_since_last_record_transformed")

# open_acc
plot_variable("open_acc")
variable_list = append(variable_list, "open_acc")


# pub_rec
variable_list = append(variable_list, "open_acc")

# revol_bal
plot_variable("revol_bal")
variable_list = append(variable_list, "revol_bal")
# revol_util - great variable
plot_variable("revol_util")
variable_list = append(variable_list, "revol_util")
# total_acc
plot_variable("total_acc")
variable_list = append(variable_list, "total_acc")

# initial_list_status
lc %>% ggplot(aes(initial_list_status, apy)) + geom_boxplot() + stat_summary(color = "red")
variable_list = append(initial_list_status, "total_acc")

# collections_12_mths_ex_med
variable_list = append(variable_list, "collections_12_mths_ex_med")

# mths_since_last_major_derog - I'm going to bin this based on whether it is NA or not
lc$mths_since_last_major_derog_transformed = "Present"
lc$mths_since_last_major_derog_transformed[is.na(lc$mths_since_last_major_derog)] = "Absent"
lc %>% ggplot(aes(mths_since_last_major_derog_transformed, apy)) + geom_boxplot() + stat_summary(color = "red")
variable_list = append(variable_list, "mths_since_last_major_derog_transformed")
# application_type
variable_list = append(variable_list, "application_type")


# tot_coll_amt - including variables like this is going to mess up our model, because it is the after-effect
# of the loan performance
# tot_cur_bal

# addr_state
us = map_data("state")
lc$addr_state = tolower(state.name[match(lc$addr_state, state.abb)])
lc %>% group_by(addr_state) %>% summarize(avg_apy = mean(apy)) %>% 
  filter(addr_state %in% unique(us$region)) %>% ggplot(aes(fill=avg_apy, map_id=addr_state)) + 
  geom_map(map=us) + 
  expand_limits(x = us$long, y = us$lat) + ylab("") + xlab("")
# let's do some feature engineering
# it looks like "Moynihan's law of the Canadian border"
# I am going to convert each state into its distance from Canada (using latitude of center of population
# as a proxy)
pop_centers = readr::read_csv("pop_center.txt")
pop_centers$STNAME = tolower(pop_centers$STNAME)
# make alaska equal washington and hawaii equal florida so they are not outliers
pop_centers$LATITUDE[pop_centers$STNAME == "alaska"] = pop_centers$LATITUDE[pop_centers$STNAME == "washington"]
pop_centers$LATITUDE[pop_centers$STNAME == "hawaii"] = pop_centers$LATITUDE[pop_centers$STNAME == "florida"]
lc = left_join(lc, pop_centers %>% select(STNAME, LATITUDE), by = c("addr_state" = "STNAME"))

# i have another hunch, let's match each state with its NAEP score (using 4th grade math 2015)
naep = readr::read_csv("naep.csv")
naep$state = tolower(naep$state)
naep = rename(naep, pisa_score = average_score)
lc = left_join(lc, naep, by = c("addr_state" = "state"))


# let's see if any of my new variables can explain the apy - doesn't look like it
ggplot(aes(x = LATITUDE, y = transformed_apy), data = lc) + geom_hex() + stat_smooth(color="red")
ggplot(aes(x = pisa_score, y = transformed_apy), data = lc) + geom_hex() + stat_smooth(color="red")
summary(lm(transformed_apy ~ LATITUDE, data = lc))
summary(lm(transformed_apy ~ pisa_score, data = lc))

