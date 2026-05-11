# FECAP - Fundação de Comércio Álvares Penteado

<p align="center">
<a href= "https://www.fecap.br/"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRhZPrRa89Kma0ZZogxm0pi-tCn_TLKeHGVxywp-LXAFGR3B1DPouAJYHgKZGV0XTEf4AE&usqp=CAU" alt="FECAP - Fundação de Comércio Álvares Penteado" border="0"></a>
</p>

---

## 🧑‍💻 Integrantes: [Flávia Costa](https://www.linkedin.com/in/flaviacostafaria/), [Guilherme Muniz](https://www.linkedin.com/in/guimuniiz/), [Lucas Moreira](https://www.linkedin.com/in/lucasmoreiragodoy/) e [Maria Eduarda](https://www.linkedin.com/in/maria-eduarda-c-foloni/)

## 🧑‍🏫 Professores Orientadores: [Marcos Minoru Nakatsugawa](https://www.linkedin.com/in/marcosminorunakatsugawa/), [Rafael Diogo Rossetti](https://www.linkedin.com/in/rafael-diogo-rossetti/), [Rodnil da Silva Moreira Lisboa](https://www.linkedin.com/in/professorrodnil/), [Rodrigo da Rosa](https://www.linkedin.com/in/rodrigo-da-rosa-phd/) e [Victor Bruno Alexander Rosetti de Quiroz](https://www.linkedin.com/in/victorbarq/)

---

# Descrição
Projeto Interdisciplinar (5º Semestre - Ciência da Computação) desenvolvido para a organização **Lideranças Empáticas (LE)**. O sistema propõe uma solução baseada em Visão Computacional e técnicas de Inteligência Artificial para automatizar o processo de triagem e contagem de doações. 

A aplicação é capaz de identificar, classificar e contar pacotes de alimentos (Arroz, Feijão, Açúcar, Macarrão, etc.) em um ambiente controlado, registrando automaticamente o volume de arrecadação por equipe e gerando evidências visuais para auditoria.

---

# Detalhes

## 💻 Tecnologias
- **[Python 3.12](https://www.python.org/)**: Linguagem base do backend e do detector de Visão Computacional.
- **[OpenCV](https://opencv.org/)**: Captura de vídeo, processamento de frames e calibração por homografia.
- **[Ultralytics / YOLOv8](https://github.com/ultralytics/ultralytics)**: Framework de IA para o treinamento e inferência do modelo de detecção.
- **[FastAPI](https://fastapi.tiangolo.com/)**: API REST do backend e gateway local do detector.
- **[SQLAlchemy 2.0 async + Alembic](https://www.sqlalchemy.org/)**: ORM assíncrono e versionamento de migrations.
- **[Next.js 14 / TanStack Start](https://tanstack.com/start)**: Frontend SPA em TypeScript + React 19.
- **[TailwindCSS + shadcn/ui](https://tailwindcss.com/)**: Sistema de design do frontend.
- **[Supabase / PostgreSQL](https://supabase.com/)**: Banco relacional gerenciado e Storage para os frames de evidência.
- **[Docker Compose](https://docs.docker.com/compose/)**: Orquestração do backend em ambiente local.
- **[Vercel](https://vercel.com/)**: Deploy contínuo do frontend (produção).

## ⚙ Funcionalidades
- Cadastro de professores e alunos, formação de grupos (mín. 4 integrantes).
- Detecção e classificação de alimentos (arroz, feijão, açúcar, macarrão, óleo, fubá) em tempo real via câmera.
- *Tracking* anti-duplicidade por cruzamento de linha + `dedup_hash` (categoria + bbox + janela de 5s).
- Cálculo automático de peso por categoria (arroz 1kg/5kg pela largura via homografia; demais com peso fixo).
- Sessões de captura (`detection_sessions`) iniciadas pelo terminal antes da contagem.
- Dashboard com ranking de grupos, totais por categoria e feed de evidências.
- Salvamento automático de *frames* anotados no Supabase Storage por item contabilizado.

---

## 🛠 Estrutura de pastas

```bash
📂 /Raiz
├── 📂 documentos
│   ├── 📂 Entrega 1
│   └── 📂 Entrega 2
├── 📂 imgs
├── 📂 src
│   ├── 📂 backend          # API FastAPI + SQLAlchemy + Alembic (Docker local)
│   ├── 📂 cv_detector      # Detector YOLO + gateway local (sempre fora de Docker)
│   └── 📂 frontend         # SPA Next.js/TanStack Start (Vercel)
├── 📄 docker-compose.yml
├── 📄 Makefile
├── 📄 .gitignore
└── 📄 README.md
```

A pasta raiz contém os seguintes diretórios e arquivos principais:

- **`README.md`**: Guia geral, tecnologias e setup macro do projeto.
- **`documentos/`**: Documentação do projeto, dividida pelas entregas do semestre.
- **`imgs/`**: Imagens utilizadas na documentação e nas referências de arquitetura.
- **`docker-compose.yml` + `Makefile`**: Orquestração do backend em Docker para desenvolvimento e apresentação.
- **`src/`**: Código-fonte, subdividido em:
  - **`backend/`** ([README](src/backend/README.md)) — API FastAPI assíncrona, regras de negócio, autenticação JWT e migrations Alembic. Lê do Supabase e serve o frontend.
  - **`cv_detector/`** ([README](src/cv_detector/README.md)) — Pipeline de IA local: dataset, treino YOLOv8, captura da webcam (`detector.py`) e gateway HTTP (`api_yolo.py`) que escreve direto no Supabase.
  - **`frontend/`** ([README](src/frontend/README.md)) — Interface web (dashboard, ranking, gestão de grupos) consumindo a API do backend.

---

## 📋 Licença
The MIT License (MIT)

---

## 🎓 Referências
- [Documentação do YOLOv8 - Ultralytics](https://docs.ultralytics.com/)
- [Documentação FastAPI](https://fastapi.tiangolo.com/)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
