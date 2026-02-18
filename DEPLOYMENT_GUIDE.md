# MWIA v2 - Linode Deployment Guide
## Modern Ward Round Intelligence Advisor — Prototype Demo

**IMPORTANT: This is a DEMONSTRATION SYSTEM using synthetic data only. NOT for clinical use.**

---

## Prerequisites

- Linode Ubuntu 24.04 server (Shared CPU $5-10/month is fine for demo)
- Domain name pointed at your Linode IP (e.g., mwia.prototype-sandbox.dev)
- OpenAI API key

---

## Step 1: Server Setup

SSH into your Linode:

```bash
ssh root@your-linode-ip
```

Update system and install dependencies:

```bash
apt update && apt upgrade -y
apt install -y python3 python3-pip python3-venv nginx certbot python3-certbot-nginx
```

Create application directory:

```bash
mkdir -p /opt/mwia
cd /opt/mwia
```

---

## Step 2: Upload Files

Transfer these files to `/opt/mwia`:
- `mwia_backend_v2.py`
- `mwia_frontend_v2.html`
- `requirements.txt`

Using SCP from your local machine:

```bash
scp mwia_backend_v2.py root@your-linode-ip:/opt/mwia/
scp mwia_frontend_v2.html root@your-linode-ip:/opt/mwia/
scp requirements.txt root@your-linode-ip:/opt/mwia/
```

---

## Step 3: Install Python Dependencies

```bash
cd /opt/mwia
pip3 install -r requirements.txt --break-system-packages
```

---

## Step 4: Configure Environment Variables

Create a `.env` file (or set in systemd service):

```bash
nano /opt/mwia/.env
```

Add:

```bash
OPENAI_API_KEY=sk-proj-YOUR_ACTUAL_KEY_HERE
MWIA_PASS_ADMIN=SecureAdminPass123!
MWIA_PASS_DOCTOR=SecureDoctorPass123!
MWIA_PASS_NURSE=SecureNursePass123!
MWIA_PASS_DEMO=SecureDemoPass123!
```

**CRITICAL:** Replace `YOUR_ACTUAL_KEY_HERE` with your real OpenAI API key.

Set proper permissions:

```bash
chmod 600 /opt/mwia/.env
chown www-data:www-data /opt/mwia/.env
```

---

## Step 5: Create Systemd Service

```bash
nano /etc/systemd/system/mwia.service
```

Paste the contents of `mwia.service` (included in deployment package), then edit:
- Replace `YOUR_KEY_HERE` with your OpenAI key
- Replace password placeholders with secure passwords

Enable and start:

```bash
systemctl daemon-reload
systemctl enable mwia
systemctl start mwia
systemctl status mwia
```

Check it's running:

```bash
curl http://localhost:8001
```

You should see HTML output.

---

## Step 6: Configure Nginx

Copy the nginx config:

```bash
cp /opt/mwia/nginx-mwia.conf /etc/nginx/sites-available/mwia
```

Edit the domain name:

```bash
nano /etc/nginx/sites-available/mwia
```

Change `mwia.prototype-sandbox.dev` to your actual domain (2 places).

Enable the site:

```bash
ln -s /etc/nginx/sites-available/mwia /etc/nginx/sites-enabled/
nginx -t  # Test configuration
systemctl reload nginx
```

---

## Step 7: Get SSL Certificate (HTTPS)

```bash
certbot --nginx -d mwia.prototype-sandbox.dev
```

Follow prompts. Certbot will automatically configure HTTPS.

---

## Step 8: Test the Deployment

Visit: `https://mwia.prototype-sandbox.dev`

You should see the login screen.

Login with your configured credentials (from Step 4).

Click **🎲 Demo** button to generate 50 test patients instantly.

---

## Maintenance

**View logs:**
```bash
journalctl -u mwia -f
```

**Restart service:**
```bash
systemctl restart mwia
```

**Update code:**
```bash
cd /opt/mwia
# Upload new files via SCP
systemctl restart mwia
```

**Monitor resource usage:**
```bash
htop
```

---

## Firewall Configuration

```bash
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 22/tcp
ufw enable
```

---

## Security Notes for Demo

✅ **Good for prototype demo:**
- HTTPS enabled
- Password-protected
- Synthetic data only
- "PROTOTYPE" watermarks everywhere

❌ **NOT production-ready:**
- In-memory storage (data lost on restart)
- No audit logging
- OpenAI API sends data externally
- No role-based access control
- No NHS IG approval

**This is a DEMONSTRATION to show capabilities and secure funding for proper development.**

---

## Troubleshooting

**Service won't start:**
```bash
journalctl -u mwia -n 50
```

**502 Bad Gateway:**
- Check mwia service is running: `systemctl status mwia`
- Check port 8001 is listening: `netstat -tlnp | grep 8001`

**OpenAI voice not working:**
- Verify API key is set correctly in systemd service
- Check logs for API errors

**CSV import fails:**
- Check file permissions: `ls -la /opt/mwia`
- Increase nginx body size if needed (already set to 20M)

---

## Cost Estimate

- **Linode Shared CPU 2GB:** $12/month
- **OpenAI API usage:** ~$5-20/month for demo (depends on usage)
- **Total:** ~$17-32/month for a live demo

---

## Next Steps After Demo Success

Once this prototype secures funding/approval:

1. Move to local LLM (no external API calls)
2. Implement proper database (PostgreSQL)
3. Add full audit logging
4. Integrate with PAS/EPR systems
5. NHS Information Governance approval process
6. Role-based access control
7. Deploy in NHS-approved environment

---

## Support

Contact: David, ICT AI Services
King's College Hospital NHS Foundation Trust

**Remember: PROTOTYPE ONLY — NOT FOR CLINICAL USE**
