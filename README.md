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
- **[Python](https://www.python.org/)**: Linguagem base para o processamento de imagens e backend.
- **[OpenCV](https://opencv.org/)**: Captura de vídeo, processamento de frames e manipulação de imagens (Data Augmentation).
- **[Ultralytics / YOLOv8](https://github.com/ultralytics/ultralytics)**: Framework de IA utilizado para o treinamento do modelo de detecção de objetos.
- **[FastAPI](https://fastapi.tiangolo.com/)**: Construção da API RESTful para comunicação entre o detector local e o servidor.
- **[Supabase / PostgreSQL](https://supabase.com/)**: Banco de dados relacional em nuvem para registro das equipes, contagens e armazenamento dos frames de evidência.

## ⚙ Funcionalidades
- Detecção e classificação de alimentos em tempo real via câmera.
- Sistema de *tracking* anti-duplicidade (contagem inteligente por cruzamento de linha no eixo Y).
- Automação de pré-processamento de dados (Geração de imagens aumentadas, divisão de treino/validação e autolabeling).
- API para recepção e validação de inferências.
- Salvamento automático de *frames* anotados (evidências) por item contabilizado.

---

## 🛠 Estrutura de pastas

```bash
📂 /Raiz
├── 📂 documentos
│   ├── 📂 Entrega 1
│   ├── 📂 Entrega 2
├── 📂 imgs
├── 📂 src
│   ├── 📂 backend
│   ├── 📂 frontend detector
│   └── 📂 frontend
├── 📄 .gitignore
├── 📄 README.md
```

A pasta raiz contém os seguintes diretórios e arquivos principais:

- **`README.md`**: Guia e explicação geral sobre o projeto, setup e tecnologias utilizadas.
- **`documentos/`**: Concentra toda a documentação do projeto, dividida pelas entregas do semestre.
- **`imgs/`**: Imagens gerais utilizadas na documentação e arquitetura do projeto.
- **`src/`**: Diretório principal do código-fonte, subdividido em:
  - **`backend/`**: Servidor da aplicação (API em FastAPI) e regras de negócio para salvar dados e imagens.
  - **`frontend detector/`**: Scripts de IA local, incluindo captura da webcam (`detector.py`), scripts de treinamento do YOLO e pipeline de processamento do dataset (`generate_dataset.py`, `split_dataset.py`).
  - **`frontend/`**: Interface para o usuário final (aplicativo/dashboard) consumir os dados consolidados.

---

## 📋 Licença
The MIT License (MIT)

---

## 🎓 Referências
- [Documentação do YOLOv8 - Ultralytics](https://docs.ultralytics.com/)
- [Documentação FastAPI](https://fastapi.tiangolo.com/)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
