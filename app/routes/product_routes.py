"""
Product listing endpoints.

Originally these read from MongoDB (a separately-seeded `products`
collection). We now serve directly from the trained catalog artifact
(the same DataFrame the recommender uses) instead, because:
  * The ML pipeline already loads/cleans/persists the full catalog as
    part of `train.py` — requiring a second, separately-seeded MongoDB
    copy of the same data is an extra operational moving part with no
    benefit at this project's scale.
  * It guarantees `/products` and `/recommendations` are always looking
    at the exact same product set (no drift between DB and model).

`app/database/database.py` is kept in the project (unused by these
routes) as ready-to-use wiring for a real future feature that needs a
live database — e.g. actual user accounts and an interactions table,
which is what would be needed to add real collaborative filtering later.
"""
from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter(tags=["products"])


@router.get("/products")
def get_products(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=24, ge=1, le=100),
    category_id: int | None = Query(default=None),
):
    catalog = request.app.state.recommender.catalog
    if category_id is not None:
        catalog = catalog[catalog["category_id"] == category_id]

    start = (page - 1) * page_size
    end = start + page_size
    page_df = catalog.iloc[start:end]

    return {
        "page": page,
        "page_size": page_size,
        "total": len(catalog),
        "products": page_df[
            ["asin", "title", "imgUrl", "price", "stars", "category_id"]
        ].to_dict(orient="records"),
    }


@router.get("/products/{asin}")
def get_product(request: Request, asin: str):
    recommender = request.app.state.recommender
    row_id = recommender.get_row_by_asin(asin)
    if row_id is None:
        raise HTTPException(status_code=404, detail=f"Product {asin} not found")
    product = recommender.catalog.iloc[row_id]
    return product[
        ["asin", "title", "imgUrl", "productURL", "price", "listPrice",
         "stars", "reviews", "category_id", "isBestSeller", "boughtInLastMonth"]
    ].to_dict()
