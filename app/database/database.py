"""
Optional MongoDB wiring.

Not used by any current route (see app/routes/product_routes.py for why:
the catalog is now served from trained artifacts). Kept here, ready to
use, for a genuine future feature that needs a live database — e.g. a
real `users` and `interactions` collection, which is what would be
required to add true collaborative filtering to this project.

Only import/use this module if RECSYS_MONGO_URL is actually set;
importing it unconditionally at module load time (as the original code
did) means the app fails to start if Mongo isn't configured, even though
nothing currently needs it.
"""
import os

from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URL = os.getenv("RECSYS_MONGO_URL")

client = AsyncIOMotorClient(MONGO_URL) if MONGO_URL else None
db = client["ecommerce"] if client is not None else None
