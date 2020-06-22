from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.requests import Request
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from starlette.routing import Route, Mount
from pathlib import Path
from book2vec import core, utils
import logging

logger = logging.getLogger(__name__)


embeddings_path = Path("./book2vec/models/embeddings.json")
try:
    with embeddings_path.open() as file_obj:
        analysis = core.Book2VecAnalysis(file_obj)
except FileNotFoundError:
    logger.error(f"Unable to find an 'embeddings.json' file at {embeddings_path.absolute()}")
    logger.info("Trying to download model files.")
    utils.download_required_files()


templates = Jinja2Templates(directory="./book2vec/templates")


async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


async def get_suggested(request: Request):
    """
    POST method endpoint to return 'returnCount' suggested books.
    """
    json_data = await request.json()
    keys = [int(x) for x in json_data["keys"]]
    return_count = int(json_data["returnCount"])
    recommendations = analysis.get_suggestions(keys)
    scores = recommendations["cumulative"].values[:return_count].tolist()
    recommendations = recommendations["cumulative"].index.values[:return_count].tolist()

    recommendations = [
        dict(
            key=book_id,
            author=analysis.index_to_metadata[book_id][1],
            title=analysis.index_to_metadata[book_id][0],
            score=f"{score:.1%}",
        )
        for book_id, score in zip(recommendations, scores)
    ]

    return JSONResponse(content=recommendations)


app = Starlette(
    debug=False,
    routes=[
        Route("/", homepage),
        Route("/api/get_suggested", get_suggested, methods=["POST"]),
        Mount("/static", app=StaticFiles(directory="./book2vec/models"), name="static"),
    ],
)
