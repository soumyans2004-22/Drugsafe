from django.db import models
from django.contrib.auth.models import User


class PredictionHistory(models.Model):

    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE
    )

    drug_a = models.CharField(max_length=200)

    drug_b = models.CharField(max_length=200)

    smiles_a = models.TextField(blank=True)

    smiles_b = models.TextField(blank=True)

    risk_level = models.CharField(max_length=50)

    confidence = models.FloatField(default=0)

    major_probability = models.FloatField(default=0)

    minor_probability = models.FloatField(default=0)

    moderate_probability = models.FloatField(default=0)

    side_effects = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.drug_a} + {self.drug_b} - {self.risk_level}"