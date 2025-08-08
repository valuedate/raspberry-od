from django.db import models

class Detection(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    label = models.CharField(max_length=100) # e.g., 'person', 'car_no_plate', 'car_plate_ABC123'
    image_path = models.CharField(max_length=255)
    camera = models.CharField(max_length=255, null=True, blank=True)
    position = models.CharField(max_length=255, null=True, blank=True)
    # Optional: Store confidence score if needed
    confidence = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"{self.label} at {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

    class Meta:
        ordering = ['-timestamp'] # Show newest detections first

