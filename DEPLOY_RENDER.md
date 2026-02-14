# Deploy On Render (Docker)

## 1. Push code to GitHub
Commit and push this project to your GitHub repository.

## 2. Create Web Service on Render
1. Open Render dashboard.
2. Click **New +** -> **Web Service**.
3. Connect your GitHub repo.
4. Render will detect `render.yaml` and Docker setup.
5. Create the service.

## 3. Set required environment variables
In Render service -> **Environment**, set:
- `MONGO_URI` (recommended: MongoDB Atlas URI)
- `MAIL_USERNAME` (optional)
- `MAIL_PASSWORD` (optional)
- `MAIL_DEFAULT_SENDER` (optional)

`SECRET_KEY` and `JWT_SECRET_KEY` are auto-generated via `render.yaml`.

## 4. Deploy
Click **Manual Deploy** -> **Deploy latest commit** (or wait for auto deploy).

## 5. Verify
Open your Render URL and test:
- `/auth/login`
- `/predict/`
- screenshot OCR prediction (Tesseract is installed inside Docker image)
