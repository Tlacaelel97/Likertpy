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

#### Likert

```python
# Configurar el gráfico indicando el grupo y la fase del estudio
Likertpy.plot_likert(
    data,
    group="G1",
    survey_number=0,
    plot_scale=Likertpy.scales.msas_G1,
    compute_percentages=True,
    bar_labels=False,
)
```

#### Moda

```python
# Configurar el gráfico indicando el grupo 
Likertpy.plot_mode(data,group="G1")
```

#### Valor Máximo

```python
# Configurar el gráfico indicando el grupo 
Likertpy.plot_max(data,group="G1")
```

#### Valor Mínimo

```python
# Configurar el gráfico indicando el grupo 
Likertpy.plot_min(data,group="G1")
```

#### Gradiente de Cambio

```python
# Configurar el gráfico indicando el grupo 
Likertpy.plot_gradient(data,group="G1")
```

## Documentación

Para más detalles sobre las funciones y configuraciones disponibles, consulta la [documentación oficial](https://github.com/Tlacaelel97/Likertpy).

## Contribuciones

Las contribuciones son bienvenidas. Si deseas colaborar, por favor sigue estos pasos:

1. Haz un fork del repositorio.
2. Crea una nueva rama con tu funcionalidad (`git checkout -b nueva-funcionalidad`).
3. Realiza cambios y súbelos (`git commit -m "Añadir nueva funcionalidad"`).
4. Envía un pull request.

## Licencia

Este proyecto está bajo la licencia MIT. Para más detalles, consulta el archivo `LICENSE`.

## Contacto

Si tienes preguntas o sugerencias, por favor abre un issue en [GitHub](https://github.com/Tlacaelel97/Likertpy/issues) o contáctanos en tlacaelel.flores@ramtechsolutions.com.mx

