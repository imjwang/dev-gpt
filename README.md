# dev-gpt
# Get started
 - ENV VARS needed:
 - GITHUB_TOKEN (fine grained)
 - GITHUB_USER (your username)
 - OPENAI_API_KEY
 - PINECONE_API_KEY
 - PINECONE_ENV
 - SERPAPI_API_KEY

```bash
cd src
pip install -r requirements.txt
export GITHUB_REPO=your-repo-name
# or
export GITHUB_REPO=movie-picker
```
### Setting up a project
I've been testing on a personal project in `src/movie-picker`. It's been pushed as a demo, but you can also clone your own repository to `/src`. The project is made for next.js + tailwindcss apps that use the pages router. It will likely fail if not a next.js project.

# Run
```bash
# this will start a conversation -> create tickets to repo -> save code to file -> push pull request
python ./main.py
# you can also assign an issue number and just get the coding part
python ./test_coder.py --issue 24
```

