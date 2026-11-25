# Captcha Solver API (ddddocr)

## Deploy
```bash
vercel
```

## Call
```bash
curl -X POST https://YOUR.vercel.app/solve \
  -H "Content-Type: application/json" \
  -d '{"image":"<BASE64>","length":6,"charset":"alnum"}'
```

## Response
```json
{"ok":true,"text":"2epgyc","engine":"ddddocr","ms":150,"votes":4}
```

Text captcha only. No reCAPTCHA / human solvers.

If Vercel build fails (size): use Render/Railway with same app.py
