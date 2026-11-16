library(lme4)

# ============================================================
# 1. Load data
# ============================================================
cat("Loading data...\n")
data <- read.csv(".../glmm_data_cong_diff_set1.csv") # Complete here the directory to the file
cat("Data loaded:", nrow(data), "rows,", ncol(data), "columns\n\n")

# ============================================================
# 2. Convert variables to appropriate types
# ============================================================
data$Subject <- factor(data$Subject)

# Binary categorical factors
data$imagine        <- factor(data$imagine, levels = c("det", "imag"))
data$congruency     <- factor(data$congruency, levels = c("incong", "cong"))
data$object_present <- factor(data$object_present, levels = c("absent", "present"))
data$scene_example <- factor(data$scene_example)  # Add scene as factor

# ============================================================
# 3. Set contrasts for sum coding
# ============================================================
contrasts(data$imagine)        <- contr.sum(2)
contrasts(data$congruency)     <- contr.sum(2)
contrasts(data$object_present) <- contr.sum(2)

cat("✓ Applied sum coding to imagine, congruency, object_present\n")

# ============================================================
# 4. Center continuous predictor
# ============================================================

cat("Subjects:", length(unique(data$Subject)), "\n")
cat("Observations:", nrow(data), "\n\n")

# ============================================================
# 5. Fit full GLMM (random slopes)
# ============================================================
cat("=== FITTING FULL MODEL ===\n")
model_full <- tryCatch({
  glmer(
    positive_answer ~ imagine  * object_present * congruency + 
      (1 + imagine + congruency + object_present | Subject), #+
    #  (1 + imagine + congruency + object_present | scene_example),
    data = data,
    family = binomial(link = "logit"),
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
  )
}, error = function(e) {
  cat("ERROR in full model:", e$message, "\n")
  return(NULL)
})

if (!is.null(model_full)) {
  cat("\n✓ Full model completed!\n")
  print(summary(model_full))
  
  if (length(model_full@optinfo$conv$lme4$messages) > 0) {
    cat("\n⚠️ CONVERGENCE WARNINGS:\n")
    print(model_full@optinfo$conv$lme4$messages)
  } else {
    cat("\n✓ Model converged successfully!\n")
  }
} else {
  cat("\n✗ Full model failed to fit.\n")
}

# ============================================================
# 6. Save results
# ============================================================
cat("\n=== SAVING RESULTS ===\n")
sink(".../glmm_results_basic_intr3_without_scenes_set1.txt") # Complete here the directory to the file output

cat("=== FULL MODEL ===\n\n")
if (!is.null(model_full)) print(summary(model_full)) else cat("Full model did not converge.\n") 

sink()
cat("\n✓ DONE!\n")

# ============================================================
# Save model objects
# ============================================================

saveRDS(model_full,
        ".../model_full_BASIC_intr3_without_scenes_SET1.rds") # Complete here the directory to the file

cat("\nSaved model objects for later comparison.\n")