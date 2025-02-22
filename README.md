# LikertPlot

**LikertPlot** es una librería en Python diseñada para la visualización y análisis de datos tipo Likert. Proporciona herramientas eficientes para procesar respuestas de encuestas y generar gráficos claros y centrados que facilitan la interpretación de los datos.

## Características

- **Procesamiento de datos Likert:** Normaliza y ajusta respuestas de encuestas para una representación equilibrada.
- **Gráficos intuitivos:** Centra visualmente los datos en torno a la respuesta neutral para facilitar el análisis.
- **Flexibilidad:** Compatible con múltiples formatos de entrada, incluyendo CSV, JSON, Excel y Parquet.
- **Configuración avanzada:** Permite personalizar etiquetas, colores y escalas de los gráficos.

## Instalación

Puedes instalar la librería directamente desde PyPI:

```bash
pip install likertplot
```

## Uso

### Carga de datos

```python
import pandas as pd
from likertplot import FileRead, ConfigurePlot

# Leer un archivo con respuestas Likert
data_reader = FileRead(folder="data", file="survey_results.csv")
df = data_reader.read_file_to_dataframe()
```

### Procesamiento y visualización

```python
# Configurar el gráfico
plot_config = ConfigurePlot()
adjusted_data, center, padded_data = plot_config._configure_rows(
    scale=[1, 2, 3, 4, 5], counts=df
)
```

## Documentación

Para más detalles sobre las funciones y configuraciones disponibles, consulta la [documentación oficial](https://github.com/tuusuario/likertplot).

## Contribuciones

Las contribuciones son bienvenidas. Si deseas colaborar, por favor sigue estos pasos:

1. Haz un fork del repositorio.
2. Crea una nueva rama con tu funcionalidad (`git checkout -b nueva-funcionalidad`).
3. Realiza cambios y súbelos (`git commit -m "Añadir nueva funcionalidad"`).
4. Envía un pull request.

## Licencia

Este proyecto está bajo la licencia MIT. Para más detalles, consulta el archivo `LICENSE`.

## Contacto

Si tienes preguntas o sugerencias, por favor abre un issue en [GitHub](https://github.com/tuusuario/likertplot/issues) o contáctanos en correo@example.com.

