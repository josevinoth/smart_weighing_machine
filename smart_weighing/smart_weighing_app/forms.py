from django import forms
from .models import Shop, Item, Price


class ShopForm(forms.ModelForm):
    class Meta:
        model = Shop
        fields = '__all__'
        widgets = {
            'shop_name': forms.TextInput(attrs={'class': 'form-control'}),
            'address': forms.Textarea(attrs={'class': 'form-control'}),
            'mobile_number': forms.TextInput(attrs={'class': 'form-control'}),
            'contact_person_name': forms.TextInput(attrs={'class': 'form-control'}),
            'email': forms.EmailInput(attrs={'class': 'form-control'}),
            'username': forms.TextInput(attrs={'class': 'form-control'}),  # ✨ added
            'password': forms.PasswordInput(attrs={'class': 'form-control'}),  # ✨ added
        }


class ItemForm(forms.ModelForm):
    class Meta:
        model = Item
        fields = '__all__'
        widgets = {
            'item_name': forms.TextInput(attrs={'class': 'form-control'}),
        }

    def clean_item_name(self):
        item_name = self.cleaned_data['item_name'].strip().lower()
        if Item.objects.filter(item_name__iexact=item_name).exists():
            raise forms.ValidationError("This item already exists!")
        return self.cleaned_data['item_name']

class PriceForm(forms.ModelForm):
    class Meta:
        model = Price
        fields = '__all__'
        widgets = {
            'shop': forms.Select(attrs={'class': 'form-control'}),
            'item': forms.Select(attrs={'class': 'form-control'}),
            'price_per_kg': forms.NumberInput(attrs={'class': 'form-control'}),
        }

class LoginForm(forms.Form):
    username = forms.CharField(max_length=255)
    password = forms.CharField(widget=forms.PasswordInput)

class PriceForm(forms.ModelForm):
    class Meta:
        model = Price
        fields = ['shop', 'item', 'price_per_kg']
