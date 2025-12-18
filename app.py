from fastapi import FastAPI, Request, HTTPException, status
from fastapi import Body, Depends
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from typing import Annotated, List
from pydantic import BaseModel, ValidationError
import uvicorn
import joblib
import pandas as pd  
import yaml 
import logging
import logging.config

# Читаем конфиг для логера
with open("logger_cfg.yaml", 'r', encoding='utf-8') as f:
    yaml_cfg = yaml.safe_load(f)

# Инициализируем логгер из yaml конфига
logging.config.dictConfig(yaml_cfg)
logger = logging.getLogger('logger')

# Инициализируем приложение FastAPI
app = FastAPI()

# Описываем классы входных и выходных данных
class MLRequest(BaseModel):
    credit_score: int
    geography: str
    gender: str
    age: int
    tenure: float
    balance: float
    num_of_products: int
    has_cr_card: int
    is_active_member: int
    estimated_salary: float

class MLResponse(BaseModel):
    answer: List[float]


# Загружаем мрдель
loaded_model = None
def load_model():
    global loaded_model
    if loaded_model is None:
        try:
            logger.info("Инициализируем модель из файла")
            loaded_model = joblib.load('model.joblib')
            logger.info("Инициализация модели успешна")
            return loaded_model
        except Exception as e:
            logger.error("Ошибка инициализации модели из файла: ", e)


def prepare_data(df):
    try:
        logger.info("Предобработка входных данных")
        # Если в tenure есть пропуски - заполним их рандомом по валидным tenure
        if df[df['tenure'].isna()].shape[0] > 0:
            # Заполняем пропуски tenure рандомными значениями из имеющихся в датасете `tenure`
            nan_idx = df[df['tenure'].isna()].index
            df.loc[nan_idx, 'tenure'] = np.random.choice(df.loc[~df['tenure'].isna(), 'tenure'].values, size=len(nan_idx))
            logger.info("Предобработка входных данных успешна")
            return df
        else:
            return df
    except Exception as e:
        logger.error("Ошибка предобработки данных: ", e)

@app.get('/')
def simple():
    return "Ready to work"

@app.post('/prediction')
def prediction(data: Annotated[List[MLRequest], Body()], model=Depends(load_model)) -> MLResponse:
   
    # Восстанавливаем DataFrame из JSON
    try:
        logger.info("Загрузка внешних данных")
        df = pd.DataFrame(
            {
                'credit_score': [x.credit_score for x in data],
                'geography': [x.geography for x in data],
                'gender': [x.gender for x in data],
                'age': [x.age for x in data],
                'tenure': [x.tenure for x in data],
                'balance': [x.balance for x in data],
                'num_of_products': [x.num_of_products for x in data],
                'has_cr_card': [x.has_cr_card for x in data],
                'is_active_member': [x.is_active_member for x in data],
                'estimated_salary': [x.estimated_salary for x in data]
            }
        )
        logger.info("Загрузка внешних данных успешна")
        # Готовим данные
        df = prepare_data(df).copy()
        # Предсказываем отток
        try:
            logger.info("Предсказание оттока по входным данным")
            predictions = model.predict(df)
            return MLResponse(answer=predictions)
        except Exception as e:
            logger.error("Ошибка формирования предсказаний: ", e)
            raise HTTPException( # Если что-то пошло не так - выдаем ошибку с кодом 500
                status_code = 500,
                detail='Ошибка формирования предсказаний'
            )
    except Exception as e:
            logger.error("Ошибка входных данных: ", e)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
    ):
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Ошибка валидации запроса",
            "errors": exc.errors(),  # Список ошибок валидации
        },
    )

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == '__main__':
    main()
