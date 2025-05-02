import datetime
import os
from django.core.management.base import BaseCommand
from django.utils import timezone
from viewer.models import Detection
from django.conf import settings

class Command(BaseCommand):
    help = 'Deletes detection records and associated image files older than a specified number of days.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--days', type=int, default=60,
            help='Delete detections older than this number of days (default: 60)'
        )

    def handle(self, *args, **options):
        days_to_keep = options["days"]
        cutoff_date = timezone.now() - datetime.timedelta(days=days_to_keep)

        self.stdout.write(f"Looking for detections older than {cutoff_date.strftime("%Y-%m-%d %H:%M:%S")}...")

        # Find old detections
        old_detections = Detection.objects.filter(timestamp__lt=cutoff_date)
        count = old_detections.count()

        if count == 0:
            self.stdout.write(self.style.SUCCESS("No old detections found to delete."))
            return

        self.stdout.write(f"Found {count} old detection records to delete.")

        deleted_files = 0
        failed_deletions = 0

        for detection in old_detections:
            # Construct absolute path to the image file
            try:
                # Ensure MEDIA_ROOT is defined in settings
                if hasattr(settings, 'MEDIA_ROOT') and settings.MEDIA_ROOT:
                    image_path_absolute = os.path.join(settings.MEDIA_ROOT, detection.image_path)
                    if os.path.exists(image_path_absolute):
                        os.remove(image_path_absolute)
                        self.stdout.write(f"Deleted image file: {image_path_absolute}")
                        deleted_files += 1
                    else:
                        self.stdout.write(self.style.WARNING(f"Image file not found, skipping deletion: {image_path_absolute}"))
                else:
                    self.stdout.write(self.style.ERROR("MEDIA_ROOT not configured in settings. Cannot delete image files."))
                    # Optionally break or continue without deleting files
                    # break
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error deleting file {detection.image_path}: {e}"))
                failed_deletions += 1

        # Delete the database records
        deleted_count, _ = old_detections.delete()

        self.stdout.write(self.style.SUCCESS(f"Successfully deleted {deleted_count} detection records from the database."))
        if hasattr(settings, 'MEDIA_ROOT') and settings.MEDIA_ROOT:
            self.stdout.write(f"Deleted {deleted_files} associated image files.")
            if failed_deletions > 0:
                self.stdout.write(self.style.WARNING(f"Failed to delete {failed_deletions} image files."))

