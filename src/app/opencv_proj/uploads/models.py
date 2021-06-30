from django.db import models
from numpy.lib.npyio import save
from .utils import get_black_key
from PIL import Image
import numpy as np
from io import BytesIO
from django.core.files.base import ContentFile
# Create your models here.

ACTION_CHOICES=(
    ('NO_FILTER', 'no filter'),
    ('BLACK_KEYS', 'black keys'),
    ('WHITE_KEYS', 'white keys'),
)

class Upload(models.Model):
    image = models.ImageField(upload_to="images")
    action = models.CharField(max_length=50, choices=ACTION_CHOICES)
    updated = models.DateTimeField(auto_now=True)
    created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.id)

    def save(self, *args, **kwargs):
        #open image
        pil_img = Image.open(self.image)

        cv_img = np.array(pil_img)
        img = get_black_key(cv_img, self.action)

        #convert back to pil
        im_pil = Image.fromarray(img)

        #save

        buffer = BytesIO()
        im_pil.save(buffer, format='png')
        image_png = buffer.getvalue()

        self.image.save(str(self.image), ContentFile(image_png), save=False)

        super().save(*args, **kwargs)