(() => {
  const applyLanguage = (language) => {
    const selected = language === 'en' ? 'en' : 'zh';
    document.documentElement.lang = selected === 'en' ? 'en' : 'zh-CN';
    document.querySelectorAll('[data-lang]').forEach((element) => {
      element.hidden = element.dataset.lang !== selected;
    });
    document.querySelectorAll('[data-set-language]').forEach((button) => {
      button.setAttribute('aria-pressed', String(button.dataset.setLanguage === selected));
    });
    const nav = document.querySelector('.journal-nav');
    if (nav) nav.setAttribute('aria-label', selected === 'en' ? 'Main navigation' : '主导航');
    try { localStorage.setItem('journal-language', selected); } catch (_) { /* Storage is optional. */ }
  };
  let language = 'zh';
  try { language = localStorage.getItem('journal-language') || 'zh'; } catch (_) { /* Use Chinese by default. */ }
  applyLanguage(language);
  document.querySelectorAll('[data-set-language]').forEach((button) => {
    button.addEventListener('click', () => applyLanguage(button.dataset.setLanguage));
  });
})();
