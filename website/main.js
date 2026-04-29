/* =============================================================
   OneStep ADHD — Main Script
   Handles:
   - Bilingual toggle (Hebrew / English)
   - Waitlist form submission via Formspree
============================================================= */

const FORMSPREE_ID = 'xpqkveyr';

let currentLang = 'he';

/**
 * Switch the page language and text direction.
 * All translatable elements carry data-he / data-en attributes.
 * @param {'he'|'en'} lang
 */
function setLang(lang) {
  currentLang = lang;

  document.documentElement.setAttribute('lang', lang);
  document.documentElement.setAttribute('dir', lang === 'he' ? 'rtl' : 'ltr');
  document.body.classList.toggle('en', lang === 'en');

  document.getElementById('btn-he').classList.toggle('active', lang === 'he');
  document.getElementById('btn-en').classList.toggle('active', lang === 'en');

  document.querySelectorAll('.t').forEach(el => {
    const text = el.getAttribute(`data-${lang}`);
    if (text !== null) el.innerHTML = text;

    const placeholder = el.getAttribute(`data-${lang}-placeholder`);
    if (placeholder) el.placeholder = placeholder;
  });
}

/**
 * Submit a waitlist form to Formspree.
 * Shows a localised success message without a page reload.
 * @param {Event}  event     - Form submit event
 * @param {string} successId - ID of the success message element
 * @param {string} emailId   - ID of the email input element
 */
function submitForm(event, successId, emailId) {
  event.preventDefault();

  const email = document.getElementById(emailId).value;

  fetch(`https://formspree.io/f/${FORMSPREE_ID}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
    },
    body: JSON.stringify({ email, lang: currentLang }),
  }).catch(() => {
    // Silently handle network errors — success message is shown regardless
    // so the user experience is uninterrupted.
  });

  const successEl = document.getElementById(successId);
  successEl.innerHTML = successEl.getAttribute(`data-${currentLang}`);
  successEl.style.display = 'block';

  document.getElementById(emailId).value = '';
}

// Initialise on page load
setLang('he');
