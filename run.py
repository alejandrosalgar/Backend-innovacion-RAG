"""
Arranque del API sin escribir uvicorn en consola.

Uso (desde la carpeta backend):
  python run.py

En Windows, el modo --reload a veces provoca WinError 10013; por defecto
reload esta desactivado en Windows (activar con PMDI_API_RELOAD=1).
Si el puerto 8080 falla (Hyper-V / rangos reservados), prueba otro, p. ej.:
  set PMDI_API_PORT=8765
"""

from __future__ import annotations

import uvicorn
from app.settings import get_settings

if __name__ == "__main__":
    settings = get_settings()

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
