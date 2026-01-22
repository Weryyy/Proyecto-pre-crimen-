# 🔧 Resolución de Conflictos Git - Guía Paso a Paso

## Problema

Tu rama `copilot/create-dashboard-with-mapbox` tiene conflictos con la rama `main` en estos archivos:
- `Project_Pre_Crime/launch_dashboard.py`
- `Project_Pre_Crime/verify_installation.py`

## ¿Por qué ocurre esto?

Los mismos archivos fueron añadidos en ambas ramas (tu rama y main) con contenido idéntico. Git detecta esto como un conflicto de tipo "both added" (ambos añadidos), aunque el contenido sea el mismo.

## ✅ La Buena Noticia

**El contenido de los archivos es 100% idéntico.** La única diferencia es que en tu rama los archivos tienen permisos de ejecución (`chmod +x`), mientras que en main no los tienen.

## 📋 Pasos para Resolver el Conflicto

### Opción 1: Resolución en la Línea de Comandos (Recomendado)

```bash
# 1. Asegúrate de estar en tu rama
git checkout copilot/create-dashboard-with-mapbox

# 2. Intenta fusionar main en tu rama
git merge main

# 3. Git mostrará conflictos, pero NO hay conflictos de contenido
# Verás algo como:
# CONFLICT (add/add): Merge conflict in Project_Pre_Crime/launch_dashboard.py
# CONFLICT (add/add): Merge conflict in Project_Pre_Crime/verify_installation.py

# 4. Marca los archivos como resueltos (esto dice a git que uses tu versión)
git add Project_Pre_Crime/launch_dashboard.py
git add Project_Pre_Crime/verify_installation.py

# 5. Completa el merge
git commit -m "Merge main into copilot/create-dashboard-with-mapbox"

# 6. Sube los cambios
git push origin copilot/create-dashboard-with-mapbox
```

### Opción 2: Resolución en GitHub (Interfaz Web)

Si prefieres usar la interfaz web de GitHub:

1. Ve a tu Pull Request en GitHub
2. Haz clic en "Resolve conflicts" (Resolver conflictos)
3. GitHub te mostrará los archivos en conflicto
4. **Como el contenido es idéntico**, simplemente acepta una de las versiones
5. Marca los conflictos como resueltos
6. Haz commit de los cambios

### Opción 3: Usar la Estrategia "Ours" (Mantener tu versión)

Si quieres mantener tu versión (con permisos ejecutables):

```bash
# Fusionar favoreciendo tu versión para estos archivos
git merge main
git checkout --ours Project_Pre_Crime/launch_dashboard.py
git checkout --ours Project_Pre_Crime/verify_installation.py
git add Project_Pre_Crime/launch_dashboard.py Project_Pre_Crime/verify_installation.py
git commit -m "Merge main, keeping executable permissions"
git push origin copilot/create-dashboard-with-mapbox
```

### Opción 4: Usar la Estrategia "Theirs" (Usar versión de main)

Si prefieres usar la versión de main (sin permisos ejecutables):

```bash
# Fusionar favoreciendo la versión de main
git merge main
git checkout --theirs Project_Pre_Crime/launch_dashboard.py
git checkout --theirs Project_Pre_Crime/verify_installation.py
git add Project_Pre_Crime/launch_dashboard.py Project_Pre_Crime/verify_installation.py
git commit -m "Merge main, using main's file permissions"
git push origin copilot/create-dashboard-with-mapbox
```

## 🔍 Verificación

Después de resolver, verifica que todo está bien:

```bash
# Ver el estado de git
git status

# Debe mostrar: "Your branch is ahead of 'origin/...' by X commits"
# y NO debe mostrar conflictos

# Ver el diff con main
git diff main

# Solo debe mostrar diferencias en permisos de archivos (si elegiste Opción 3)
```

## 📝 Explicación Técnica

**¿Qué pasó?**
1. Tu rama añadió estos archivos en el commit `24fd71c` (Add installation verification script)
2. La rama main añadió los mismos archivos en el commit `1adb5e5` (Add Mapbox visualization...)
3. Cuando GitHub intenta fusionar, detecta que ambas ramas añadieron los mismos archivos
4. Git marca esto como conflicto "add/add" aunque el contenido sea idéntico

**¿Por qué no hay conflictos de contenido?**
El contenido de los archivos es 100% igual. La única diferencia es:
- Tu rama: `chmod +x` (permisos ejecutables -rwxr-xr-x)
- Main: sin `chmod +x` (permisos normales -rw-r--r--)

## 💡 Recomendación

**Usa la Opción 1** (resolución manual con `git add`). Es la más simple y mantiene tu versión con permisos ejecutables, lo cual es correcto para scripts Python que se ejecutan directamente.

## ❓ ¿Necesitas Ayuda?

Si tienes problemas:
1. Copia el error exacto que ves
2. Copia la salida de `git status`
3. Pregunta en el PR y te ayudaré

## 📚 Recursos Adicionales

- [Git Merge Conflicts](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts)
- [Resolving a merge conflict using the command line](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/addressing-merge-conflicts/resolving-a-merge-conflict-using-the-command-line)

---

**Resumen:** Los archivos son idénticos en contenido. Solo necesitas decirle a git que acepte una de las versiones usando `git add` y luego hacer commit. ✅
